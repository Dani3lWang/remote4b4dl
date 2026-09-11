import torch
import torch.nn as nn
from vtimellm.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from abc import ABC, abstractmethod

class VTimeLLMMetaModel:

    def initialize_vision_modules(self, model_args):
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # Frame positions are an optional B4DL extension.  Keep the default
        # disabled so checkpoints produced before this feature retain their
        # original architecture and inference behaviour.
        use_frame_position_embedding = getattr(
            model_args,
            'use_frame_position_embedding',
            getattr(self.config, 'use_frame_position_embedding', False),
        )
        frame_position_max = getattr(
            model_args,
            'frame_position_max',
            getattr(self.config, 'frame_position_max', 64),
        )
        frame_position_max = int(frame_position_max)
        if frame_position_max <= 0:
            raise ValueError(
                f"frame_position_max must be positive, got {frame_position_max}"
            )
        self.config.use_frame_position_embedding = bool(
            use_frame_position_embedding
        )
        self.config.frame_position_max = frame_position_max

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(768, self.config.hidden_size)

        if self.config.use_frame_position_embedding:
            if not hasattr(self, 'frame_position_embedding'):
                self.frame_position_embedding = nn.Embedding(
                    frame_position_max, self.config.hidden_size
                )
                # Zero initialization makes the pre-training forward pass
                # exactly match the B3 baseline before position learning.
                nn.init.zeros_(self.frame_position_embedding.weight)
            elif self.frame_position_embedding.num_embeddings != frame_position_max:
                raise ValueError(
                    "Existing frame_position_embedding has "
                    f"{self.frame_position_embedding.num_embeddings} entries, "
                    f"but frame_position_max={frame_position_max}."
                )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print("load mlp:", pretrain_mm_mlp_adapter)


class VTimeLLMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        frame_indices=None,
    ):
        # print(position_ids, attention_mask)
        # if past_key_values:
        #     print(past_key_values[-1][-1].shape)
        # print(input_ids.shape, position_ids.shape, attention_mask.shape, past_key_values.shape, images)
        if images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                if hasattr(past_key_values, 'get_seq_length'):
                    target_shape = past_key_values.get_seq_length() + 1
                elif self.get_model().config.model_type == 'chatglm':
                    target_shape = past_key_values[-1][-1].shape[0] + 1
                else:
                    target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.get_model().mm_projector(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            # image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.get_model().mm_projector(images)

        # Normalize both supported image layouts to a per-sample list.  The
        # collator uses a list for variable-length scenes, while inference
        # commonly supplies a [B, N, 768] tensor.
        if not isinstance(image_features, (list, tuple)):
            if image_features.ndim == 2:
                image_features = image_features.unsqueeze(0)
            image_features = list(image_features)

        if getattr(self.config, 'use_frame_position_embedding', False):
            if frame_indices is None:
                raise ValueError(
                    "frame_indices is required when "
                    "use_frame_position_embedding=True"
                )
            if torch.is_tensor(frame_indices):
                if frame_indices.ndim == 1:
                    frame_indices = [frame_indices]
                elif frame_indices.ndim == 2:
                    frame_indices = list(frame_indices)
                else:
                    raise ValueError(
                        "frame_indices tensor must have shape [N] or [B, N], "
                        f"got {tuple(frame_indices.shape)}"
                    )
            else:
                frame_indices = list(frame_indices)

            if len(frame_indices) != len(image_features):
                raise ValueError(
                    "frame_indices batch size does not match image features: "
                    f"{len(frame_indices)} vs {len(image_features)}"
                )

            position_embedding = self.get_model().frame_position_embedding
            max_position = position_embedding.num_embeddings
            positioned_features = []
            for sample_idx, (features, indices) in enumerate(
                zip(image_features, frame_indices)
            ):
                indices = torch.as_tensor(
                    indices, dtype=torch.long, device=features.device
                )
                if indices.ndim != 1:
                    raise ValueError(
                        f"frame_indices[{sample_idx}] must be one-dimensional, "
                        f"got {tuple(indices.shape)}"
                    )
                if indices.numel() != features.shape[0]:
                    raise ValueError(
                        f"frame_indices[{sample_idx}] has {indices.numel()} entries "
                        f"for {features.shape[0]} feature frames"
                    )
                if indices.numel() > 0 and (
                    int(indices.min()) < 0 or int(indices.max()) >= max_position
                ):
                    raise ValueError(
                        f"frame_indices[{sample_idx}] must be in [0, {max_position - 1}]"
                    )
                positions = position_embedding(indices).to(dtype=features.dtype)
                positioned_features.append(features + positions)
            image_features = positioned_features
        # print([image.shape for image in image_features])
        
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if self.get_model().config.model_type == 'chatglm':
            fake_input_ids = torch.full((new_input_embeds.shape[0], new_input_embeds.shape[1]), -10000, 
                                        dtype=new_input_embeds.dtype, device=new_input_embeds.device)
            attention_mask = attention_mask.to(torch.int8)
            new_input_embeds = new_input_embeds.transpose(0, 1).contiguous()
        else:
            fake_input_ids = None
        # print(position_ids, attention_mask)
        return fake_input_ids, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
