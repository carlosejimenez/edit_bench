
class MixtralTensorNames(ArchitectureInfo, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "MixtralForCausalLM"
    num_local_experts: int

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return MixtralTensorNames(num_local_experts=config.num_local_experts)

    def pre_weights(self) -> List[str]:
        return MISTRAL_INFO.pre_weights()

    def post_weights(self) -> List[str]:
        return MISTRAL_INFO.post_weights()

    def embed_weights(self) -> List[str]:
        return MISTRAL_INFO.embed_weights()

    def num_layers_config_key(self) -> str:
        return MISTRAL_INFO.num_layers_config_key()

    def layer_weight_formats(self) -> List[str]:
        num_experts = self.num_local_experts
        res = [fmt for fmt in MISTRAL_INFO.layer_weight_formats() if ".mlp." not in fmt]
        for expert_idx in range(num_experts):
            for param in ("w1", "w2", "w3"):
                fmt = (
                    MISTRAL_INFO.layer_prefix_format
                    + f".block_sparse_moe.experts.{expert_idx}.{param}.weight"
                )
                res.append(fmt)
        res.append(MISTRAL_INFO.layer_prefix_format + ".block_sparse_moe.gate.weight")
        return res
