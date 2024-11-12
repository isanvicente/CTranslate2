import argparse
import json
import os

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, transformer_spec


_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELU,
    "fast_gelu": common_spec.Activation.GELUTanh,
    "relu": common_spec.Activation.RELU,
    "silu": common_spec.Activation.SWISH,
}

_SUPPORTED_FEATURES_MERGE = {
    "concat": common_spec.EmbeddingsMerge.CONCAT,
    "sum": common_spec.EmbeddingsMerge.ADD,
}


def check_opt(opt, num_source_embeddings):
    with_relative_position = getattr(opt, "max_relative_positions", 0) > 0
    with_rotary = getattr(opt, "max_relative_positions", 0) == -1
    with_alibi = getattr(opt, "max_relative_positions", 0) == -2
    activation_fn = getattr(opt, "pos_ffn_activation_fn", "relu")
    feat_merge = getattr(opt, "feat_merge", "concat")
    self_attn_type = getattr(opt, "self_attn_type", "scaled-dot")

    check = utils.ConfigurationChecker()
    check(
        opt["model"]["encoder"]["encoder_type"] == opt["model"]["decoder"]["decoder_type"]
        and opt["model"]["decoder"]["decoder_type"] in {"transformer", "transformer_lm"},
        "Options --encoder_type and --decoder_type must be"
        " 'transformer' or 'transformer_lm",
    )
    check(
        self_attn_type == "scaled-dot",
        "Option --self_attn_type %s is not supported (supported values are: scaled-dot)"
        % self_attn_type,
    )
    check(
        activation_fn in _SUPPORTED_ACTIVATIONS,
        "Option --pos_ffn_activation_fn %s is not supported (supported activations are: %s)"
        % (activation_fn, ", ".join(_SUPPORTED_ACTIVATIONS.keys())),
    )
    check(
        opt["model"]["embeddings"]["position_encoding"] != (with_relative_position or with_rotary or with_alibi),
        "Options --position_encoding and --max_relative_positions cannot be both enabled "
        "or both disabled",
    )
    check(
        num_source_embeddings == 1 or feat_merge in _SUPPORTED_FEATURES_MERGE,
        "Option --feat_merge %s is not supported (supported merge modes are: %s)"
        % (feat_merge, " ".join(_SUPPORTED_FEATURES_MERGE.keys())),
    )
    check.validate()


def _get_model_spec_seq2seq(
    opt, variables, src_vocabs, tgt_vocabs, num_source_embeddings
):
    """Creates a model specification from the model options."""
    with_relative_position = opt.get("max_relative_positions", 0) > 0
    activation_fn = opt.get("pos_ffn_activation_fn", "relu")
    feat_merge = opt.get("feat_merge", "concat")

    # Return the first head of the last layer unless the model was trained with alignments.
    if opt.get("lambda_align", 0) == 0:
        alignment_layer = -1
        alignment_heads = 1
    else:
        alignment_layer = opt.get("alignment_layer")
        alignment_heads = opt.get("alignment_heads")

    num_heads = opt.get("heads", 8)

    enc_layers = opt.get("enc_layers", -1)
    if enc_layers<0:
        enc_layers=opt.get("layers")
    dec_layers = opt.get("dec_layers", -1)
    if dec_layers<0:
        dec_layers=opt.get("layers")

    model_spec = transformer_spec.TransformerSpec.from_config(
        (enc_layers, dec_layers),
        num_heads,
        with_relative_position=with_relative_position,
        activation=_SUPPORTED_ACTIVATIONS[activation_fn],
        alignment_layer=alignment_layer,
        alignment_heads=alignment_heads,
        num_source_embeddings=num_source_embeddings,
        embeddings_merge=_SUPPORTED_FEATURES_MERGE[feat_merge],
        multi_query_attention=opt.get("multiquery", False),
    )

    model_spec.config.decoder_start_token = opt.get("decoder_start_token", "<s>")

    set_transformer_spec(model_spec, variables)
    for src_vocab in src_vocabs:
        model_spec.register_source_vocabulary(src_vocab)
    for tgt_vocab in tgt_vocabs:
        model_spec.register_target_vocabulary(tgt_vocab)

    return model_spec


def _get_model_spec_lm(opt, variables, src_vocabs, tgt_vocabs, num_source_embeddings):
    """Creates a model specification from the model options."""
    with_relative_position = getattr(opt, "max_relative_positions", 0) > 0
    with_rotary = getattr(opt, "max_relative_positions", 0) == -1
    with_alibi = getattr(opt, "max_relative_positions", 0) == -2
    activation_fn = getattr(opt, "pos_ffn_activation_fn", "relu")
    num_heads = getattr(opt, "heads", 8)
    num_kv = getattr(opt, "num_kv", 0)
    if num_kv == num_heads or num_kv == 0:
        num_kv = None
    rotary_dim = 0 if with_rotary else None
    rotary_interleave = getattr(opt, "rotary_interleave", True)
    ffn_glu = activation_fn == "silu"
    sliding_window = getattr(opt, "sliding_window", 0)

    model_spec = transformer_spec.TransformerDecoderModelSpec.from_config(
        opt.dec_layers,
        num_heads,
        activation=_SUPPORTED_ACTIVATIONS[activation_fn],
        ffn_glu=ffn_glu,
        with_relative_position=with_relative_position,
        alibi=with_alibi,
        rms_norm=opt.layer_norm == "rms",
        rotary_dim=rotary_dim,
        rotary_interleave=rotary_interleave,
        multi_query_attention=getattr(opt, "multiquery", False),
        num_heads_kv=num_kv,
        sliding_window=sliding_window,
    )

    model_spec.config.layer_norm_epsilon = getattr(opt, "norm_eps", 1e-6)

    set_transformer_decoder(
        model_spec.decoder,
        variables,
        with_encoder_attention=False,
    )

    for tgt_vocab in tgt_vocabs:
        model_spec.register_vocabulary(tgt_vocab)

    return model_spec


def get_vocabs(vocab):
    if isinstance(vocab, dict) and "src" in vocab:
        if isinstance(vocab["src"], list):
            src_vocabs = [vocab["src"]]
            tgt_vocabs = [vocab["tgt"]]

            src_feats = vocab.get("src_feats")
            if src_feats is not None:
                src_vocabs.extend(src_feats.values())
        else:
            src_vocabs = [field[1].vocab.itos for field in vocab["src"].fields]
            tgt_vocabs = [field[1].vocab.itos for field in vocab["tgt"].fields]
    else:
        # Compatibility with older models.
        src_vocabs = [vocab[0][1].itos]
        tgt_vocabs = [vocab[1][1].itos]

    return src_vocabs, tgt_vocabs


class EoleConverter(Converter):
    """Converts models generated by eole."""

    def __init__(self, model_path: str):
        """Initializes the eole converter.

        Arguments:
          model_path: Path to the eole safetensor model (directory containing 'model.00.safetensors', 'vocab.json', 'config.json' and 'optimizer.pt files').
        """
        self._model_path = model_path

    def _load(self):
        import torch
        from safetensors.torch import load_file

        checkpoint = None
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"{self._model_path} does not seem to exist.")
        elif os.path.isdir(self._model_path):
            os.environ["MODEL_PATH"] = self._model_path            
            checkpoint = {}
            config_path = os.path.join(self._model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config_dict = json.loads(os.path.expandvars(f.read()))
                    # drop data to prevent validation issues
                    config_dict["data"] = {}
                    # drop inference to prevent validation issues
                    if "inference" in config_dict.keys():
                        config_dict.pop("inference")
                    if "training" in config_dict.keys():
                        config_dict["training"]["dummy_load"] = True
                    else:
                        config_dict["training"] = {"dummy_load": True}
                    #_config = TrainConfig(**config_dict)
                    checkpoint["opt"] = config_dict
            else:
                raise FileNotFoundError(f"{self._model_path} does not contain config.json")
            vocab_path = os.path.join(self._model_path, "vocab.json")
            if os.path.exists(vocab_path):
                with open(vocab_path) as f:
                    checkpoint["vocab"] = json.load(f)
                # use default specials if not specified
                if "specials" not in checkpoint["vocab"].keys():
                    checkpoint["vocab"]["specials"] = {
                        "bos_token": DefaultTokens.BOS,
                        "pad_token": DefaultTokens.PAD,
                        "eos_token": DefaultTokens.EOS,
                        "unk_token": DefaultTokens.UNK,
                    }
            else:
                raise FileNotFoundError(f"{self._model_path} does not contain vocab.json")
            optim_path = os.path.join(self._model_path, "optimizer.pt")
            if os.path.exists(optim_path):
                checkpoint["optim"] = torch.load(
                    optim_path, map_location=torch.device("cpu")
                )
        else:
            raise FileNotFoundError(f"{self._model_path} is not a directory.")

        checkpoint["model"]=load_file(os.path.join(self._model_path,'model.00.safetensors'))
        src_vocabs, tgt_vocabs = get_vocabs(checkpoint["vocab"])

        check_opt(checkpoint["opt"], num_source_embeddings=len(src_vocabs))
        import pdb
        pdb.set_trace()
        variables = checkpoint["model"]
        ## the following code is not needed because generator layers are included in the safetensor file.
        # variables.update(
        #     {
        #         "generator.%s" % key: value
        #         for key, value in checkpoint["generator"].items()
        #     }
        # )

        if checkpoint["opt"]["model"]["decoder"]["decoder_type"] == "transformer_lm":
            return _get_model_spec_lm(
                checkpoint["opt"]["model"],
                variables,
                src_vocabs,
                tgt_vocabs,
                num_source_embeddings=len(src_vocabs),
            )
        else:
            return _get_model_spec_seq2seq(
                checkpoint["opt"]["model"],
                variables,
                src_vocabs,
                tgt_vocabs,
                num_source_embeddings=len(src_vocabs),
            )


def set_transformer_spec(spec, variables):
    set_transformer_encoder(spec.encoder, variables)
    set_transformer_decoder(spec.decoder, variables)


def set_transformer_encoder(spec, variables):
    set_input_layers(spec, variables, "encoder")
    set_layer_norm(spec.layer_norm, variables, "encoder.layer_norm")
    for i, layer in enumerate(spec.layer):
        # from onmt encoder.transformer.n to eole's encoder_transformer_layers.n
        set_transformer_encoder_layer(layer, variables, "encoder.transformer_layers.%d" % i)


def set_transformer_decoder(spec, variables, with_encoder_attention=True):
    set_input_layers(spec, variables, "decoder")
    set_layer_norm(spec.layer_norm, variables, "decoder.layer_norm")
    for i, layer in enumerate(spec.layer):
        set_transformer_decoder_layer(
            layer,
            variables,
            "decoder.transformer_layers.%d" % i,
            with_encoder_attention=with_encoder_attention,
        )

    try:
        set_linear(spec.projection, variables, "generator")
    except KeyError:
        # Compatibility when the generator was a nn.Sequential module.
        set_linear(spec.projection, variables, "generator.0")


def set_input_layers(spec, variables, scope):
    if hasattr(spec, "position_encodings"):
        # mapping embeddings (*.embedding.make_embedding.pe) to *_emb.pe.pe
        if scope == "encoder":
            emb="src_emb.pe"
        else:
            emb="tgt_emb.pe"

        set_position_encodings(
            spec.position_encodings,
            variables,
            emb,
            #"%s.embeddings.make_embedding.pe" % scope,
        )
    else:
        # See https://github.com/OpenNMT/OpenNMT-py/issues/1722
        spec.scale_embeddings = False

    embeddings_specs = spec.embeddings
    # # ONMT version could contain various embedding sources. Not eole 
    # if not isinstance(embeddings_specs, list):
    #     embeddings_specs = [embeddings_specs]
    ## encoder embeddings are stored in a list (onmt legacy)
    if isinstance(embeddings_specs, list):
        embeddings_specs = embeddings_specs[0]
    
    # mapping embeddings weights (*.embeddings.make_embedding.emb_luts.*) to *_emb.embeddings.weight
    if scope == "encoder":
        embw="src_emb.embeddings"
    else:
        embw="tgt_emb.embeddings"

    set_embeddings(
            embeddings_specs,
            variables,
            embw,
    )

    """
    for i, embeddings_spec in enumerate(embeddings_specs):
        set_embeddings(
            embeddings_spec,
            variables,
            "%s.embeddings.make_embedding.emb_luts.%d" % (scope, i),
        )
    """

def set_transformer_encoder_layer(spec, variables, scope):
    # feed_forward is now mlp
    set_ffn(spec.ffn, variables, "%s.mlp" % scope)
    set_layer_norm(spec.ffn.layer_norm, variables, "%s.input_layernorm" % scope)
    set_multi_head_attention(
        spec.self_attention,
        variables,
        "%s.self_attn" % scope,
        self_attention=True,
    )
    set_layer_norm(spec.self_attention.layer_norm, variables, "%s.post_attention_layernorm" % scope)


def set_transformer_decoder_layer(spec, variables, scope, with_encoder_attention=True):
    # feed_forward is now mlp
    set_ffn(spec.ffn, variables, "%s.mlp" % scope)
    set_layer_norm(spec.ffn.layer_norm, variables, "%s.input_layernorm" % scope)
    set_multi_head_attention(
        spec.self_attention,
        variables,
        "%s.self_attn" % scope,
        self_attention=True,
    )

    # post_attention_layernorm = layer_norm_1
    set_layer_norm(spec.self_attention.layer_norm, variables, "%s.post_attention_layernorm" % scope)

    if with_encoder_attention:
        #set_layer_norm(spec.attention.layer_norm, variables, "%s.precontext_layernorm" % scope)
        set_multi_head_attention(spec.attention, variables, "%s.context_attn" % scope)
        # post_attention_layernorm = layer_norm_2
        set_layer_norm(spec.attention.layer_norm, variables, "%s.precontext_layernorm" % scope)
       
    

def set_ffn(spec, variables, scope):
    # set_layer_norm(spec.layer_norm, variables, "%s.layer_norm" % scope)
    # w_1 = gate_up_proj
    set_linear(spec.linear_0, variables, "%s.gate_up_proj" % scope)
    # w_2 = gate_down_proj
    set_linear(spec.linear_1, variables, "%s.down_proj" % scope)
    # # commented out for eole for the time being (2024/11/11)
    # if hasattr(spec, "linear_0_noact"):
    #     set_linear(spec.linear_0_noact, variables, "%s.w_3" % scope)


def set_multi_head_attention(spec, variables, scope, self_attention=False):
    if self_attention:
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        set_linear(split_layers[0], variables, "%s.linear_query" % scope)
        set_linear(split_layers[1], variables, "%s.linear_keys" % scope)
        set_linear(split_layers[2], variables, "%s.linear_values" % scope)
        utils.fuse_linear(spec.linear[0], split_layers)
    else:
        set_linear(spec.linear[0], variables, "%s.linear_query" % scope)
        split_layers = [common_spec.LinearSpec() for _ in range(2)]
        set_linear(split_layers[0], variables, "%s.linear_keys" % scope)
        set_linear(split_layers[1], variables, "%s.linear_values" % scope)
        utils.fuse_linear(spec.linear[1], split_layers)
    set_linear(spec.linear[-1], variables, "%s.final_linear" % scope)
    if hasattr(spec, "relative_position_keys"):
        spec.relative_position_keys = _get_variable(
            variables, "%s.relative_positions_embeddings.weight" % scope
        )
        spec.relative_position_values = spec.relative_position_keys


def set_layer_norm(spec, variables, scope):
    try:
        spec.gamma = _get_variable(variables, "%s.weight" % scope)
    except KeyError:
        # Compatibility with older models using a custom LayerNorm module.
        sys.stderr.write(f"layer weight not found: {scope}.weight")
        pass
    try:
        spec.beta = _get_variable(variables, "%s.bias" % scope)
    except KeyError:
        pass


def set_linear(spec, variables, scope):
    spec.weight = _get_variable(variables, "%s.weight" % scope)
    bias = variables.get("%s.bias" % scope)
    if bias is not None:
        spec.bias = bias


def set_embeddings(spec, variables, scope):
    spec.weight = _get_variable(variables, "%s.weight" % scope)


def set_position_encodings(spec, variables, scope):
    spec.encodings = _get_variable(variables, "%s.pe" % scope).squeeze()


def _get_variable(variables, name):
    return variables[name]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", required=True, help="Model path.")
    Converter.declare_arguments(parser)
    args = parser.parse_args()
    EoleConverter(args.model_path).convert_from_args(args)


if __name__ == "__main__":
    main()
