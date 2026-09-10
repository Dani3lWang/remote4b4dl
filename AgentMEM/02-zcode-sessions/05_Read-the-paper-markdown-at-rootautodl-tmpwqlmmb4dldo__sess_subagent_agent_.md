# Read the paper markdown at /root/autodl-tmp/wql/mmb4dl/do...

| 项 | 值 |
|---|---|
| 会话 ID | `sess_subagent_agent_b957f5b2-adc1-486b-8742-0e613dc69cf5` |
| 工作目录 | `/root/autodl-tmp/wql/mmb4dl` |
| 父会话 | `sess_ac5b7d2d-de7b-4d89-b54e-4c4508f244a3` |
| 时间 | 2026-08-24 00:42 → 2026-08-24 00:45 |
| 模型 | GLM-5.3 |
| 消息数 | 7（文本块 5，工具调用 8） |

---

### ASSISTANT  ·  `GLM-5.3`

> 🔀 模型切换：None → builtin:bigmodel-start-plan/GLM-5.3


### USER

Read the paper markdown at /root/autodl-tmp/wql/mmb4dl/docs/mmb4dl-md/mmb4dl.md (a full ACM MM 2025 paper "B4DL: A Benchmark for 4D LiDAR LLM in Spatio-Temporal Understanding"). Extract a PRECISE implementation spec covering everything the paper states about the model, training, and evaluation. I need exact quotes + line numbers for each item. Specifically extract:

1. **Architecture (Section 4.1 + Figure 4/5/6)**: 
   - The 4D LiDAR encoder: what architecture (e.g., LiDAR-CLIP?), how each frame P_t is encoded, input format (points per frame, channels), how frames are aggregated into S_L, any positional/temporal encoding
   - The LiDAR-Text Aligner: exact structure (linear layers? MLP? Q-former?), input/output dims if stated
   - Metatoken: exact formulation (Appendix C), what metadata fields (velocity/direction/position/acceleration relative to previous frames), which frames get descriptions, exact connective phrases
   - The base LLM used (Vicuna? version? 7B?)
2. **Input text format**: exact token order (<4DLiDAR>, <video>, question, <meta>...) from Appendix C and Figure 6, how frame numbers appear in questions/answers ("from frame 004 to frame 006" format)
3. **Training (Section 4.2 + 5.1)**: two stages, what data each stage uses, what is frozen vs trainable in each stage (encoder? aligner? LLM? embeddings?), loss, optimizer, learning rate, batch size, epochs, GPU, total time
4. **Evaluation (Section 5)**: test set details, the 6 tasks and which metrics apply to which task, how mIoU is defined for time grounding (exact formula), how accuracy is computed for simple tasks, which caption metrics (BLEU-4, ROUGE-L, METEOR, BERTScore, GPT-score) on which subset, baseline comparisons (B4DL-LiDARLLM, VTimeLLM)
5. **Dataset construction details (Section 3)**: sequence segmentation (6 sequences per scene, 3-10 frames, 2-keyframe interval), per-sequence QA generation (40 samples per sequence distribution across 6 tasks), frame numbering convention
6. **Any hyperparameters in appendix** (batch size, lr, epochs, LoRA config if any)

Report as a structured checklist with exact quotes and markdown line numbers. Be thorough — this is the ground-truth spec for a code alignment audit. Do NOT summarize loosely; quote precisely.


### ASSISTANT  ·  `GLM-5.3`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/docs/mmb4dl-md/mmb4dl.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# B4DL: A Benchmark for 4D LiDAR LLM in Spatio-Temporal Understanding
2	
3	Changho Choi*
4	
5	Korea Advanced Institute of Science
6	
7	and Technology
8	
9	Daejeon, Republic of Korea
10	
11	ccho4702@kaist.ac.kr
12	
13	Youngwoo Shin*
14	
15	Korea Advanced Institute of Science
16	
17	and Technology
18	
19	Daejeon, Republic of Korea
20	
21	yshin0917@kaist.ac.kr
22	
23	Gyojin Han
24	
25	Korea Advanced Institute of Science
26	
27	and Technology
28	
29	Daejeon, Republic of Korea
30	
31	hangj0820@kaist.ac.kr
32	
33	Dong-Jae Lee
34	
35	Korea Advanced Institute of Science
36	
37	and Technology
38	
39	Daejeon, Republic of Korea
40	
41	jhtwosun@kaist.ac.kr
42	
43	Junmo Kim
44	
45	Korea Advanced Institute of Science
46	
47	and Technology
48	
49	Daejeon, Republic of Korea
50	
51	junmo.kim@kaist.ac.kr

…（已截断，原文 75988 字符）
```

</details>

<details><summary>思考过程</summary>

The user wants me to read a paper markdown file and extract a precise implementation spec with exact quotes and line numbers. Let me start by reading the file.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: wc -l /root/autodl-tmp/wql/mmb4dl/docs/mmb4dl-md/mmb4dl.md
description: Count lines in paper markdown
```

</details>

<details><summary>输出摘要</summary>

```
436 /root/autodl-tmp/wql/mmb4dl/docs/mmb4dl-md/mmb4dl.md
```

</details>


### ASSISTANT  ·  `GLM-5.3`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/docs/mmb4dl-md/images/5_945_234_687_444_0.jpg
```

</details>

<details><summary>输出摘要</summary>

```
[Attached image/jpeg: Read image]
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/docs/mmb4dl-md/images/15_197_234_1406_599_0.jpg
```

</details>

<details><summary>输出摘要</summary>

```
[Attached image/jpeg: Read image]
```

</details>

<details><summary>思考过程</summary>

I've read the entire file. Now let me compile the precise implementation spec with exact quotes and line numbers. The file is 436 lines. Let me organize everything carefully.

Let me also check the Figure 6 image to extract any details about the metatoken and input format — but the images are local. Let me check if I can view the image for Figure 6 to get more detail about the connective phrases. Actually, I can read image files with the Read tool. Let me look at Figure 6 (line 419-421: images/15_197_234_1406_599_0.jpg) and Figure 4 (line 207: images/5_945_234_687_444_0.jpg) since these are called out explicitly in the request. The user asked about "Figure 4/5/6" and exact connective phrases from Figure 6.

Let me read those images.

</details>

I have read the full paper. Let me also examine the key figures (Figure 4 training pipeline and Figure 6 metatoken construction) since the user asked for details from them, including exact connective phrases.


### ASSISTANT  ·  `GLM-5.3`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/docs/mmb4dl-md/images/6_301_640_1199_650_0.jpg
```

</details>

<details><summary>输出摘要</summary>

```
[Attached image/jpeg: Read image]
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/docs/mmb4dl-md/images/17_138_236_1535_1543_0.jpg
```

</details>

<details><summary>输出摘要</summary>

```
[Attached image/jpeg: Read image]
```

</details>

<details><summary>思考过程</summary>

The images were uploaded to CDN. Let me analyze them with the vision model to extract the details from Figure 4 and Figure 6.

</details>

<details><summary>思考过程</summary>

Let me get a more detailed transcription of Figure 6, especially the red-highlighted connective phrases. The vision model gave a partial transcription. Let me try again with a more focused prompt on the connective phrases and the metatoken text.

</details>

<details><summary>思考过程</summary>

The vision model gave slightly conflicting red-highlight details between two calls. The paper text at line 375 says: "These are joined using the red-highlighted connective phrases shown in Figure 6." The figure shows the <meta> template: "<meta> The metadata of the first frame is 'The ego vehicle is ...' and the metadata of the last frame is 'The ego vehicle is ...'". The red-highlighted connective phrases are likely "The metadata of the first frame is", "and the metadata of the last frame is". Let me do one more verification pass on the figure to confirm which exact phrases are red, and also check the frame numbering in the sample question ("from frame 10 to frame 22" — note: Figure 6 shows "frame 10 to frame 22", NOT the "frame 010 to frame 022" zero-padded format; the prompts use "from frame 000 to frame 000" format).

Also, let me check Figure 5 (qualitative comparison) and Figures 7/9/10 for sample QA formats with frame numbers. Actually, the user asked specifically about Figure 4/5/6 for architecture. Figure 5 is the qualitative comparison. Let me quickly check Figure 5 image for the input format details (e.g., "<video>" token for VTimeLLM).

Also, let me check Figure 9/10 images to see the exact QA formats with frame numbering. That could help with the "from frame 004 to frame 006" format question. The user asked about "how frame numbers appear in questions/answers ('from frame 004 to frame 006' format)". The paper's Table 8 prompt says: 'Include temporal information such as the corresponding frame number using the format "from frame 000 to frame 000".' And Table 9 GPT scoring example says "from frame 004 to frame 006" and "frame 0 to frame 8", "frame 00 to frame 16", "frame 026". Table 10 example says "between Frame 015 and Frame 016" and "From frames 12 to 20". Figure 6 sample question says "from frame 10 to frame 22".

So there is inconsistency in the paper itself, which is worth noting in the spec. Let me quickly examine Figure 5 and maybe Figure 9 for the QA formats.

Let me also verify the red highlights in figure 6 with a different prompt approach, and check figure 5 (image 6_301_640_1199_650_0.jpg) and figure 9 (17_138_236_1535_1543_0.jpg).

</details>

The images are available via CDN. Let me analyze them to extract the figure details.**🌐 Z.ai Built-in Tool: analyze_image**

**Input:**
```json
{"imageSource":"https://maas-log-prod.cn-wlcb.ufileos.com/anthropic/b957f5b2-adc1-486b-8742-0e613dc69cf5/5_945_234_687_444_0.jpg?UCloudPublicKey=TOKEN_e15ba47a-d098-4fbd-9afc-a0dcf0e4e621&Expires=1787505168&Signature=0esaSiIQjaAtIjL4EvYE+AZWdEE=","prompt":"This is Figure 4 from a paper about the B4DL model (4D LiDAR MLLM). It shows the training pipeline with a 4D LiDAR encoder, LiDAR aligner (projection layer f_p), LoRA module, LLM (Vicuna). Transcribe ALL text in the figure exactly, including: module names, arrows, frozen/trainable indicators, training stage labels, input formats (tokens like <4DLiDAR>, <meta>), any dimensions, and captions. Be exhaustive and literal."}
```
*Executing on server...*
**🌐 Z.ai Built-in Tool: analyze_image**

**Input:**
```json
{"imageSource":"https://maas-log-prod.cn-wlcb.ufileos.com/anthropic/b957f5b2-adc1-486b-8742-0e613dc69cf5/15_197_234_1406_599_0.jpg?UCloudPublicKey=TOKEN_e15ba47a-d098-4fbd-9afc-a0dcf0e4e621&Expires=1787505168&Signature=OcV0b+CPEvHGOZldkcr760AUEfw=","prompt":"This is Figure 6 from the B4DL paper showing \"Visualization of metatoken construction and a sample B4DL input. Raw numbers are converted to text, with the first and last frame texts joined by red-highlighted words to form the metatoken.\" Transcribe ALL text in the figure EXACTLY and completely: the raw metadata numbers/tables (velocity, direction, position, acceleration, timestamp), the textual descriptions they are converted into for the first frame and last frame, the red-highlighted connective phrases joining them, and the full sample input sequence including tokens like <4DLiDAR> and <meta> and the question. Be exhaustive and literal, preserving exact wording."}
```
*Executing on server...*
**Output:**
**analyze_image_result_summary:** [{"text": "\"Textual Description for 3D LiDAR\\nLarge Language Model\\nProjection Layer\\n(fₚ)\\nLiDAR Encoder\\n(Eₗ)\\nText Input\\n3D LiDAR Understanding Stage\\nTextual Description for 4D LiDAR\\nLarge Language Model\\nLoRA\\nProjection Layer\\n(fₚ)\\nLiDAR Encoder\\n(Eₗ)\\nText Input\\nMetatoken...
                                                **Output:**
**analyze_image_result_summary:** [{"text": "\"What can be seen on the right side of the road from frame 10 to frame 22?\\n<4DLiDAR> What can be ... frame 10 to frame 22?\\n<meta> The metadata of the first frame is ‘The ego vehicle is ...’ and the metadata of the last frame is ‘The ego vehicle is ...’\\ncalibrated_sensor:\\n{'transl...
                                                Let me get a more precise transcription of the connective phrases in Figure 6.**🌐 Z.ai Built-in Tool: analyze_image**

**Input:**
```json
{"imageSource":"https://maas-log-prod.cn-wlcb.ufileos.com/anthropic/b957f5b2-adc1-486b-8742-0e613dc69cf5/15_197_234_1406_599_0.jpg?UCloudPublicKey=TOKEN_e15ba47a-d098-4fbd-9afc-a0dcf0e4e621&Expires=1787505168&Signature=OcV0b+CPEvHGOZldkcr760AUEfw=","prompt":"Look at the section of this figure labeled \"Input for the B4DL model\" and the \"<meta>\" line. Transcribe character-by-character the full <meta> sentence template, including every word, quote marks, and ellipses. I need the exact template with placeholders, e.g. \"The metadata of the first frame is '...' and the metadata of the last frame is '...'\". Also transcribe the two sentences describing the ego vehicle state (first frame text and last frame text) as completely as possible, and identify which specific words/phrases are highlighted in red."}
```
*Executing on server...*
**Output:**
**analyze_image_result_summary:** [{"text": "\"The full <meta> sentence template is:  \\n`<meta> The metadata of the first frame is '...' and the metadata of the last frame is '...'`  \\n\\nThe two sentences describing the ego vehicle state (first frame and last frame) are:  \\n- First frame: `The ego vehicle is ...`  \\n- Last fram...
                                                The vision tool gave slightly inconsistent red-highlight reports between calls. Let me verify the exact red phrases once more, and also check Figure 5 and Figure 9/10 for exact QA input formats and frame numbering.


### ASSISTANT  ·  `GLM-5.3`

<details><summary>思考过程</summary>

**🌐 Z.ai Built-in Tool: analyze_image**

**Input:**
```json
{"imageSource":"https://maas-log-prod.cn-wlcb.ufileos.com/anthropic/b957f5b2-adc1-486b-8742-0e613dc69cf5/6_301_640_1199_650_0.jpg?UCloudPublicKey=TOKEN_e15ba47a-d098-4fbd-9afc-a0dcf0e4e621&Expires=1787505226&Signature=m0Lsg4D0Wbun6AYRzekQ/vQXrf4=","prompt":"This is Figure 5 from the B4DL paper: qualitative comparison of generated answers from B4DL-LiDARLLM (a) and VTimeLLM (b). Transcribe ALL text exactly, especially the questions shown (note any tokens like <video> or frame references such as 'from frame 012 to frame 018') and the model answers shown for each model. Be exhaustive and literal."}
```
*Executing on server...*
**🌐 Z.ai Built-in Tool: analyze_image**

**Input:**
```json
{"imageSource":"https://maas-log-prod.cn-wlcb.ufileos.com/anthropic/b957f5b2-adc1-486b-8742-0e613dc69cf5/17_138_236_1535_1543_0.jpg?UCloudPublicKey=TOKEN_e15ba47a-d098-4fbd-9afc-a0dcf0e4e621&Expires=1787505226&Signature=TjW6hErJSpHwpt/aEZEcg4uGYXo=","prompt":"This is Figure 9 from the B4DL paper showing extra examples of the generated dataset for 6 different tasks (Existence, Binary QA, Time Grounding, Description, Temporal Understanding, Comprehensive Reasoning). Transcribe every question and answer pair in the figure exactly as written, preserving exact frame number formats (e.g., 'from frame 004 to frame 006', 'frame 0', 'frame 10') and any special tokens."}
```
*Executing on server...***Output:**
**analyze_image_result_summary:** [{"text": "\"Figure 5(a): Question: Which area does the black car near the pedestrian crossing move toward from frame 012 to frame 018? B4DL-LiDARLLM: [Full image of scene 1] In this scene, there are various objects such as buildings, parked cars, traffic cones, and a pedestrian crossing area. Near the ped...
                                                **Output:**
**analyze_image_result_summary:** [{"text": "\"Existence\\nQ: Are there any cars in the scene during the period from frame 0 to frame 7?\\nA: Yes, there are multiple cars in the scene during this period.\\nBinary QA\\nQ: Is the top of the pedestrian bridge in the back view visible from frame 0 to frame 4?\\nA: Yes, the top of the p...
                                                Figure 9 confirms the frame numbering formats vary in the dataset examples ("frame 0 to frame 7", "from frame 004 to frame 006", "Frame 0 to Frame 8"). I now have all the information needed. Let me also double-check whether there are any hyperparameters mentioned elsewhere (search for "learning rate", "batch", "epoch", "AdamW", "lr") to be thorough.

</details>

Both figures are transcribed. Let me do a final grep sweep for any hyperparameter mentions I might have missed (learning rate, batch size, epochs, optimizer, LoRA rank, etc.).

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n -i -E "learning rate|batch|epoch|adamw|optimizer|lr|rank|768|1024|4096|hidden|dimension|freeze|frozen|trainable|hours|GPU|4090|A100|Vicuna|ViT|CLIP" /root/autodl-tmp/wql/mmb4dl/docs/mmb4dl-md/
…（已截断，原文 209 字符）
description: Search paper for hyperparameter and architecture keywords
```

</details>

<details><summary>输出摘要</summary>

```
61:due to the absence of high-quality, modality-specific annotations and the lack of MLLM architectures capable of processing its high-dimensional composition. To address these challenges, we introduce B4DL, a new benchmark specifically designed for training and evaluating MLLMs on 4D LiDAR understanding. In addition, we propose a scalable data generation pipeline and an MLLM model that, for the first time, directly processes raw 4D LiDAR by bridging it with language understanding. Combined with our dataset and benchmark, our model offers a unified solution for spatio-temporal reasoning in dynamic outdoor environments. We provide rendered 4D LiDAR videos, generated dataset, and inference outputs on diverse scenarios at: https://github.com/ccho4702/B4DL
87:Although the 4D LiDAR [17, 22] off
…（已截断，原文 13874 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n -i -E "frame [0-9]|frame \{|IoU|intersection|union|accuracy|exact|metric" /root/autodl-tmp/wql/mmb4dl/docs/mmb4dl-md/mmb4dl.md | head -60
description: Search paper for frame number formats and metric definitions
```

</details>

<details><summary>输出摘要</summary>

```
119:To construct a large-scale dataset for these tasks, we utilize nuScenes [3] as the data source. Our data generation pipeline leverages multi-view camera images as intermediaries to extract language-expressive 4D LiDAR context using GPT-40 [1], and complements this with structured human annotations as the ground truth. This pipeline ensures both rich linguistic expressiveness and accurate spatio-temporal grounding. We further present task-specific evaluation metrics and detailed dataset statistics to highlight the scale and diversity of the benchmark.
129:Evaluation metrics are set to align with the objectives of each task type and to fairly assess the model's performance. For the Simple Tasks, we use accuracy for Existence and Binary QA, evaluated based on exact answer matching, and mI
…（已截断，原文 21048 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3`

<details><summary>思考过程</summary>

I now have everything. The paper does NOT state: mIoU formula explicitly (only "mIoU for Time Grounding to measure overlap between predicted and ground-truth temporal segments"), no learning rate, no batch size, no epochs, no optimizer name (only cross-entropy loss for causal language modeling), no LoRA rank/config details. These are genuinely absent — important to note as gaps for the code audit.

Let me compile the final structured checklist with exact quotes and line numbers. Line numbers refer to /root/autodl-tmp/wql/mmb4dl/docs/mmb4dl-md/mmb4dl.md.

Note discrepancies I found:
- Line 157: "the training set comprises approximately 14.8k samples, and the testing set includes 30.1k samples" — this contradicts Table 2 (148,271 train / 30,145 test) and the abstract (178.4k). "14.8k" appears to be a typo for 148.2k. Worth flagging.
- Line 93 vs 161: module named "LiDAR-Text Aligner" in intro, "LiDAR Aligner" in Section 4.
- Line 139: "two descriptions per sequence" (front group and rear group). Note Table 7 prompt actually uses front/front_left/front_right (3 views) for the "frontal group".
- Frame numbering inconsistencies: Table 8 prompt says "from frame 000 to frame 000" format; Table 9 examples show "from frame 004 to frame 006", "frame 0 to frame 8", "frame 00 to frame 16", "frame 026"; Table 10 shows "between Frame 015 and Frame 016" and "From frames 12 to 20"; Figure 6 sample question: "from frame 10 to frame 22"; Figure 9 examples: "frame 0 to frame 7", "from frame 004 to frame 006", "Frame 0 to Frame 8".
- Figure 6: metatoken input format "<4DLiDAR> {question} <meta> The metadata of the first frame is '...' and the metadata of the last frame is '...'" — vision analysis; red-highlighted connectives are "The metadata of the first frame is", "and the metadata of the last frame is" (two calls disagreed on red; the connective "and" joining is the most consistent reading, but I should present with the caveat that the vision transcription of red-highlight is uncertain between calls; the load-bearing content is the exact template text).
- No "<video>" token appears anywhere in the paper — the user asked about it, so I must report it's absent (VTimeLLM's input format is not described).
- Line 187: "we include textual descriptions only for the initial and final frames" — and Appendix C refines: "the metatoken descriptions of the first and last frames referenced in the QA pair".

Also encoder details:
- Line 165: encoder E_L, CLIP-aligned, "Inspired by LiDARCLIP [9], we employ an encoder that leverages voxelization and sensor calibration to ensure consistent spatial alignment across frames."
- Line 171: similarity loss Eq 1: L_similarity = (1/d)(z_I - z_L)^T (z_I - z_L), z_I, z_L ∈ R^d, positive pairs only.
- Line 174: Eq 2: S_L = {P_1,...,P_T}, P_t ∈ R^{N_t × 4} (x,y,z + intensity).
- Line 180: Eq 3: {p_i^cls, p_i^1, ..., p_i^k} = E_L(P_i), i=1..T.
- Line 183: "The global embeddings p_i^cls across all frames are concatenated to form the 4D LiDAR Embedding" — so only cls tokens used, not the k local features.
- Line 185: Aligner f_p "a single linear layer".
- No positional/temporal encoding mentioned — absent.
- Line 205: Vicuna-7b-v1.5, frozen; encoder pre-trained on nuScenes; image encoder CLIP ViT-L/14.

Training:
- Stage 1 (3D): LiDAR-LLM-Nu-Caption dataset [34] with 162k QA pairs; only f_p trainable, everything else frozen (incl. E_L, LLM).
- Stage 2 (4D): B4DL dataset 178k QA; LoRA module introduced, "freezing all other components, including the encoder E_L and the LiDAR aligner f_p"; '<4DLiDAR>' prepended to all QA inputs.
- Loss: "cross-entropy loss for causal language modeling" (line 161).
- Hardware: "single NVIDIA RTX 4090 GPU and complete within 24 hours" (line 211).
- No LR, batch size, epochs, optimizer, LoRA rank anywhere.

Evaluation:
- Line 129: metrics per task; line 213: "each metric is computed per task and then averaged to obtain the final score. Specifically, the accuracy of Simple Tasks is the average of Existence and Binary QA scores. mIoU is based on the Time Grounding task, while the remaining four metrics come from Complex Tasks."
- No explicit mIoU formula — just "mIoU for Time Grounding to measure overlap between predicted and ground-truth temporal segments" (line 129).
- Accuracy: "evaluated based on exact answer matching" (line 129).
- GPT-score: "reference-free evaluator to score coherence, relevance, and correctness on a 100-point scale" (line 129), prompt in Table 9 (line 413).
- Test set: 150 scenes, 900 sequences, 30,145 test annotations (Table 2, line 151); line 157.
- Baselines: B4DL-LiDARLLM (line 217: "a variant of the B4DL model trained only on the single-frame LiDAR-LLM dataset [34]"), VTimeLLM [12] (video-based MLLM). Table 3 numbers (line 223).
- Waymo † row: evaluated on 1k-sample Waymo subset (line 241).

Dataset:
- Line 155: 850 scenes (700 train / 150 test); six sequences per scene; 3-10 frames; interval of two keyframes = 1 second at 2Hz; 4200 train / 900 test sequences.
- Table 2 (line 149-151): max seq length 19, min 5; task counts. Note: sequences 3-10 frames but Max Sequence Length 19 / Min 5 — another paper inconsistency worth flagging (Table 2 says min 5 max 19; text says 3 to 10 frames).
- Line 157: 40 samples per sequence: 5 Existence, 10 Binary QA, 5 Time Grounding, 5 Scene Description, 5 Temporal Understanding, 10 Comprehensive Reasoning.
- Line 147: post-processing; "time grounding samples explicitly include phrases like 'from frame A to frame B'".
- Line 139: six synchronized multi-view images (front, front-right, front-left, back, back-right, back-left) per frame; split into frontal/rear groups → two descriptions per sequence.

Appendix hyperparameters: none. Appendix E scaling, Appendix C metatoken, Appendix F prompts.

Now let me write the final report. It should be a structured checklist with exact quotes + line numbers. Line numbers must be accurate. Let me double check a few line numbers:
- Line 165: encoder paragraph. Yes.
- Line 167-169: Eq 1.
- Line 171: z_I, z_L ∈ R^d.
- Line 173-175: Eq 2.
- Line 177: "where S_L denotes a sequence of LiDAR frames over time, and each frame P_t contains N_t points with spatial coordinates (x, y, z) and intensity."
- Line 179-181: Eq 3.
- Line 183: p_i^cls concatenation.
- Line 185: aligner.
- Line 187: metatoken.
- Line 189-199: Section 4.2.
- Line 205: Vicuna.
- Line 211: datasets + GPU.
- Line 213: eval aggregation.
- Line 217: baselines.
- Line 219: B4DL-LiDARLLM mIoU note.
- Line 221/223: Table 3.
- Line 375: Appendix C.
- Line 405: Table 7 prompt.
- Line 409: Table 8 prompt.
- Line 413: Table 9 GPT scoring.
- Line 417: Table 10 example.
- Line 421: Figure 6 caption.

For figure transcriptions (from vision), I'll clearly mark them as [FIGURE TRANSCRIPTION — from image, not markdown text] since line numbers point to captions only.

Now write the final answer. Make it thorough and structured. No emojis. Include the caveats/gaps section.

</details>

I have completed a full read of the paper (436 lines of markdown), transcription of Figures 4, 5, 6, and 9 via vision analysis, and a keyword sweep for hyperparameters. Below is the ground-truth spec. All line numbers refer to `/root/autodl-tmp/wql/mmb4dl/docs/mmb4dl-md/mmb4dl.md`. Items marked **[FIGURE]** are transcribed from the figure images (path noted), since the markdown only contains their captions. Items marked **[NOT STATED]** are genuinely absent from the paper — important gaps for the audit.

---

# B4DL Paper Implementation Spec (ACM MM 2025)

## 1. Architecture (Sec 4.1, lines 163–187; Fig 4 caption line 209)

### 1.1 Overall composition
- Three modules (line 161): "The model consists of three key modules-Encoder for LiDAR Point Cloud, LiDAR Aligner, and Metatoken-which together allow the language model to process 4D LiDAR data in its high-dimensional form."
- Naming inconsistency to note: intro (line 93) calls it "LiDAR-Text Aligner"; Sec 4.1 (line 185) and line 161 call it "LiDAR Aligner".
- Loss (line 161): "Our architecture builds on the pretrained large language model and integrates textual and 4D LiDAR modalities using cross-entropy loss for causal language modeling, facilitating joint training over both modalities."
- Figure 4 caption (line 209): "Visualization of the training pipeline. The projection layer \( f_p \) and a LoRA module are trained in two stages respectively, while other components remain frozen. The 4D LiDAR input is marked with a red dotted box."

### 1.2 4D LiDAR Encoder E_L
- LiDAR-CLIP style, CLIP-aligned (line 165): "Encoder for LiDAR Point Cloud, dened as \( E_L \), transforms 3D LiDAR point clouds into embeddings aligned with textual representations via CLIP [25]." and "Inspired by LiDARCLIP [9], we employ an encoder that leverages voxelization and sensor calibration to ensure consistent spatial alignment across frames."
- Encoder pretraining loss, Eq (1) (lines 167–171): `\u0581` — exact markdown: `{\mathcal{L}}_{\text{ similarity }} = \frac{1}{d}{\left( {z}_{I} - {z}_{L}\right) }^{T}\left( {{z}_{I} - {z}_{L}}\right) \tag{1}` with `where \( {z}_{I},{z}_{L} \in  {\mathbb{R}}^{d} \)` (line 171), and "Unlike the contrastive loss in CLIP training, this formulation uses only positive pairs, allowing us to leverage CLIP's prior knowledge without the overhead of contrastive sampling." (line 171). Note: this loss is for pretraining E_L, paired with CLIP image encoder ViT-L/14 (line 205).
- Input format, Eq (2) (line 174): `S_L = {P_1, P_2, ..., P_T}, P_t ∈ R^{N_t × 4}`; (line 177): "where \( S_L \) denotes a sequence of LiDAR frames over time, and each frame \( P_t \) contains \( N_t \) points with spatial coordinates \( (x, y, z) \) and intensity." So channels = (x, y, z, intensity). N_t (points per frame) is a per-frame variable; no fixed point count is given. **[NOT STATED: downsampling/point count, voxel size, range crop]**
- Per-frame encoding, Eq (3) (line 180): `\{ p_i^{cls}, p_i^1, ..., p_i^k \} = E_L(P_i), i = 1, 2, ..., T`; (line 183): "where \( p_i^{cls} \) captures global spatial information, and \( \{p_i^1, ..., p_i^k\} \) represent localized features."
- Aggregation into S_L / 4D embedding (line 183): "The global embeddings \( p_i^{cls} \) across all frames are concatenated to form the 4D LiDAR Embedding, which serves as the input modality for the B4DL model." — i.e., only the CLS tokens are concatenated; the k local features are discarded for the LLM input.
- Temporal/positional encoding for the frame sequence: **[NOT STATED — no temporal or positional embedding is mentioned anywhere; aggregation is plain concatenation of per-frame CLS embeddings]**
- Image branch for pretraining (line 205): "For the image encoder, we adopt the CLIP model, specifically the ViT-L/14 variant [25]." and "we pre-train the encoder \( E_L \) using nuScenes, which provides 3D point clouds paired with multi-view images."

### 1.3 LiDAR-Text Aligner f_p
- Exact structure (line 185): "LiDAR Aligner, \( f_p \), is implemented as a single linear layer that refines encoder-generated embeddings for seamless integration with LLMs." Also: "The aligner \( f_p \) maintains the high-dimensional structure of the input while projecting it into a space compatible with language-based reasoning, thereby functioning as a key interface between vision-language features and the LLM input space."
- Input/output dims: **[NOT STATED]** (implicitly: CLIP-dim embeddings in, LLM text-embedding dim out; no numbers given). Not a Q-former, not an MLP — a single linear layer.

### 1.4 Metatoken (Sec 4.1 line 187; Appendix C line 375; Figure 6 line 421)
- Sec 4.1 (line 187): "Metatoken is introduced to incorporate sensor metadata into the model. Denoted as '<meta>', it precedes a textual description of the ego vehicle equipped with the LiDAR sensor. Since LLMs struggle with raw decimal values due to tokenizer limitations, we convert sensor data into relative direction, position, velocity, and acceleration with respect to previous frames. This enables the model to capture temporal dynamics and causality. To avoid redundancy across sequences, we include textual descriptions only for the initial and final frames, which are concatenated after the '<meta>' token."
- Metadata fields (line 375): "each LiDAR point cloud is associated with raw metadata representing the calibrated sensor's positional state, the ego vehicle's motion, and a timestamp indicating time progression. Since these values follow a consistent numerical format, we convert them into textual descriptions that capture velocity, direction, position, and acceleration relative to previous frames."
- Which frames get descriptions — refined in Appendix C (line 375): "the metatoken descriptions of the first and last frames referenced in the QA pair are concatenated." (i.e., first and last frame referenced in the QA, not simply of the sequence).
- Figure 6 caption (line 421): "Visualization of metatoken construction and a sample B4DL input. Raw numbers are converted to text, with the first and last frame texts joined by red-highlighted words to form the metatoken."
- **[FIGURE 6 transcription — `images/15_197_234_1406_599_0.jpg`]:** raw metadata shown includes `calibrated_sensor: {'translation': ..., 'rotation': ...}`, ego_pose (translation/rotation), velocity and timestamp columns. The input template shown:
  - Question: "What can be seen on the right side of the road from frame 10 to frame 22?"
  - Model input: `<4DLiDAR> What can be seen on the right side of the road from frame 10 to frame 22? <meta> The metadata of the first frame is 'The ego vehicle is ...' and the metadata of the last frame is 'The ego vehicle is ...'`
  - Connective phrases: "The metadata of the first frame is '...'" and "and the metadata of the last frame is '...'" (the joining words "The metadata of the first frame is" / "and the metadata of the last frame is" are the red-highlighted connectives; vision transcription across calls consistently gave this template text, with minor ambiguity about exactly which words are red — the literal template string above is reliable).
  - Note the figure uses non-zero-padded "frame 10 to frame 22".

### 1.5 Base LLM
- Line 205: "We adopt Vicuna-7b-v1.5 [39] as a large language model, which remains frozen during both training and inference. It serves as the backbone LLM throughout the entire pipeline."

## 2. Input text format

- Token order (Appendix C, line 375): "Specifically, the input question is prepended with <4DLiDAR>, followed by <meta>, after which the metatoken descriptions of the first and last frames referenced in the QA pair are concatenated." → order: `<4DLiDAR> {question} <meta> {first-frame description} {connective} {last-frame description}`. Confirmed by Figure 6 transcription in 1.4.
- `<4DLiDAR>` token purpose (line 197): "During training, the token '<4DLiDAR>' is prepended to all QA inputs to explicitly indicate that the task involves reasoning over 4D LiDAR data, guiding the model to attend to inherent spatiotemporal relationships within the input."
- `<video>` token: **[NOT STATED — no `<video>` token appears anywhere in the paper; the visual token fed to the LLM is the concatenated CLS 4D LiDAR Embedding (line 183), there is no named placeholder token like `<4DLiDAR>`-per-frame described]**
- Frame-number phrase requirement for Time Grounding (line 147): "For example, time grounding samples explicitly include phrases like \"from frame A to frame B\" to accurately localize events."
- Frame-number format inconsistency across the paper (audit-relevant):
  - Table 8 QA-generation prompt (line 409): "- Include temporal information such as the corresponding frame number using the format \"from frame 000 to frame 000\"." (zero-padded 3 digits, literal placeholder)
  - Table 8 also: "This is a description of the front parts of the ego vehicle from frame \{start_index\} to frame \{end_index\}" (line 409)
  - Table 9 GPT-scoring examples (line 413): "from frame 004 to frame 006", "from frame 000 to frame 008", "from frame 0 to frame 8", "from frame 00 to frame 16", "by frame 026" — mixed padding.
  - Table 10 example (line 417): "between Frame 015 and Frame 016" (capital F, zero-padded), "From frames 12 to 20" (lowercase plural, no padding), plus "Start Index: 12 / End Index: 20".
  - **[FIGURE 6]**: "from frame 10 to frame 22" (no padding).
  - **[FIGURE 9 transcription — `images/17_138_236_1535_1543_0.jpg`]**: Existence Q: "Are there any cars in the scene during the period from frame 0 to frame 7?"; Time Grounding uses "from frame 004 to frame 006"; another example "Frame 0 to Frame 8".

## 3. Training (Sec 4.2 lines 189–199; Sec 5.1 lines 203–211)

### Stage 1 — 3D LiDAR Understanding Stage
- Goal (line 193): "3D LiDAR Understanding Stage focuses on training the LiDAR aligner \( f_p \) to enable the LLM to interpret LiDAR data by projecting the point cloud embedding \( p_i^{cls} \) into the LLM's text embedding space. Although \( f_p \) is trained using only static 3D point clouds, the projection layer enables it to process them even in a concatenated form. As such, this projection layer is also trained and optimized to handle 4D LiDAR embeddings for the next stage."
- Trainable vs frozen (line 195): "During this phase, only \( f_p \) remain trainable, while all other modules, including the encoder \( E_L \), are kept frozen." (LLM also frozen per line 205.) Also: "Note that the LiDAR-LLM dataset is paired with the point cloud embeddings to focus on the spatial information contained in the LiDAR data."
- Data (line 211): "For the \( 3D \) LiDAR Understanding Stage, we use the LiDAR-LLM-Nu-Caption dataset [34], which contains 162k QA pairs based on static LiDAR frames, to capture spatial relationships in point clouds." (also line 195: "training is conducted using the LiDAR-LLM dataset, which consists of textual descriptions paired with static 3D point clouds from nuScenes")

### Stage 2 — 4D LiDAR Understanding Stage
- Trainable vs frozen (line 197): "To focus on temporal reasoning, we introduce a LoRA module [11], while freezing all other components, including the encoder \( E_L \) and the LiDAR aligner \( f_p \)." So: LoRA (on LLM) trainable; E_L, f_p, LLM base weights all frozen.
- Data (lines 199, 211): B4DL dataset, 178k QA pairs; "The model is trained on both Simple Tasks and Complex Tasks as introduced in Section 3.1" (line 199). Metatoken is incorporated in this stage (line 199: "we introduce the term 'frame' to denote specific time steps and incorporate Metatoken, previously described in Section 4.1, to provide context on the ego vehicle's motion."). Ablation Table 4 (line 239) confirms Metatoken is toggled "during the 4D LiDAR Understanding Stage" (line 243).
- Data split for both stages (line 211): "Both datasets are derived from nuScenes and follow the same data split described in Section 3.3, with 700 scenes for training and 150 for testing. This consistent split prevents test data leakage."

### Loss / optimizer / schedule / hardware
- Loss (line 161): "cross-entropy loss for causal language modeling" — the only loss stated for B4DL training (Eq 1 similarity loss is for encoder pretraining only).
- Hardware/time (line 211): "All experiments are run on a single NVIDIA RTX 4090 GPU and complete within 24 hours."
- **[NOT STATED: learning rate, optimizer (no AdamW mention), batch size, epochs, warmup, weight decay, LoRA rank/alpha/target modules, deepspeed/precision]** — grep for these terms returned nothing.

## 4. Evaluation (Sec 3.1 line 129; Sec 5 lines 211–253)

- Test set (lines 155, 151): 150 test scenes → 900 test sequences → 30,145 test annotations (Table 2: Existence 3,770; Yes/No Q&A 7,525; Time Grounding 2,783; Description 3,770; Temporal Understanding 4,757; Comprehensive Reasoning 7,540; total 30,145).
- Metric-to-task mapping (line 129): "For the Simple Tasks, we use accuracy for Existence and Binary QA, evaluated based on exact answer matching, and mIoU for Time Grounding to measure overlap between predicted and ground-truth temporal segments. For the Complex Tasks, including Description, Temporal Understanding, and Comprehensive Reasoning, we employ BLEU-4 (B@4) [23], METEOR [2], ROUGE-L [16], and BERTScore [5] to evaluate both linguistic quality and semantic relevance. We also use GPT-40 [1] as a reference-free evaluator to score coherence, relevance, and correctness on a 100-point scale."
- Aggregation (line 213): "For evaluation, each metric is computed per task and then averaged to obtain the final score. Specifically, the accuracy of Simple Tasks is the average of Existence and Binary QA scores. mIoU is based on the Time Grounding task, while the remaining four metrics come from Complex Tasks."
- mIoU definition: **no explicit formula given**. Only line 129's "mIoU for Time Grounding to measure overlap between predicted and ground-truth temporal segments". Frame intervals are the segments (answers contain "from frame A to frame B"); no intersection/union equation appears in the paper.
- Accuracy computation: "evaluated based on exact answer matching" (line 129) for Existence + Binary QA, then averaged (line 213).
- GPT-score: reference-free, 0–100 scale (line 129); the full judge prompt is Table 9 (line 413), scoring rubric: 100 fully aligned; 80–99 mostly; 60–79 partial; 40–59 limited; 20–39 barely; 0–19 unrelated; ends with "Only provide the score without any additional text." Template: `Question: \{q\} GT: \{ref\} Answer: \{pred\}`.
- Baselines (line 217): "First, to highlight the temporal advantages of 4D LiDAR over traditional 3D LiDAR, we compare our model with B4DL-LiDARLLM, a variant of the B4DL model trained only on the single-frame LiDAR-LLM dataset [34]. Second, to assess the spatial reasoning capability, we compare our model with VTimeLLM, a video-based MLLM capable of temporal reasoning."
- B4DL-LiDARLLM mIoU handling (line 219): "B4DL-LiDARLLM, which lacks temporal reasoning capability, does not generate time-related responses or frame intervals in a way that satisfies the requirements for mIoU evaluation." (reported as "-" in Table 3).
- Main results, Table 3 (line 223): B4DL-LiDARLLM: Acc 0.611, mIoU -, B@4 0.018, ROUGE-L 0.189, METEOR 0.187, BERTScore 0.868, GPT 42.425; VTimeLLM: 0.694 / 0.160 / 0.083 / 0.305 / 0.262 / 0.893 / 55.654; B4DL (Ours, Waymo-dagger): 0.706 / 0.294 / 0.053 / 0.269 / 0.228 / 0.886 / 58.248; B4DL (Ours, nuScenes): 0.762 / 0.311 / 0.095 / 0.322 / 0.275 / 0.897 / 59.513. Dagger footnote (line 221): "denotes evaluation on the Waymo dataset to assess cross-dataset generalization of the B4DL model trained solely on our B4DL training set constructed from nuScenes" — the Waymo eval is "a 1k-sample subset of Waymo without any re-training or adaptation" (line 241).
- Ablations: Table 4 (HA x Metatoken, line 239); Table 6 (training scale 10/25/50/75/100%, line 387); Table 5 (DCScore diversity, line 251).

## 5. Dataset construction (Sec 3, lines 135–157)

- Sequence segmentation (line 155): "The B4DL dataset ... is built from 850 scenes sourced from nuScenes, with 700 scenes allocated for training and 150 for testing. ... Each scene is further segmented into six sequences containing 3 to 10 frames, sampled at a consistent interval of two keyframes, which corresponds to one second given the 2Hz LiDAR frame rate. Consequently, the dataset comprises 4,200 sequences for training and 900 for testing, each paired with LiDAR point clouds, synchronized multi-view images, and numerically grounded manual annotations."
- Inconsistency to flag: Table 2 (line 151) reports "Max Sequence Length 19 / Min Sequence Length 5", while the text says "3 to 10 frames". Also line 157 says "the training set comprises approximately 14.8k samples, and the testing set includes 30.1k samples" — 14.8k contradicts Table 2's 148,271 train (likely a typo for 148.2k).
- Per-sequence QA distribution (line 157): "For each sequence, we generate 40 data samples to ensure a balanced distribution across all six reasoning tasks. Specifically, each sequence yields 5 samples for Existence, 10 for Binary QA, 5 for Time Grounding, 5 for Scene Description, 5 for Temporal Understanding, and 10 for Comprehensive Reasoning, which are subsequently refined through the post-processing step." (Table 2 caption line 149: "each sequence yields 40 samples across six tasks".)
- Context extraction (line 139): "we use six synchronized multi-view images (front, front-right, front-left, back, back-right, back-left) per frame from nuScenes ... Each image is temporally aligned with the corresponding LiDAR frame captured at the same timestamp. ... the views are split into frontal and rear groups, resulting in two descriptions per sequence."
- Note: the actual Table 7 description prompt (line 405) uses only three views: "The first \{frame_len\} frames are from the front view, the next \{frame_len\} from the front_left, and the last \{frame_len\} from the front_right" — i.e., the "frontal group" = front/front_left/front_right; the back group is analogous (prompt shown is the frontal one).
- Human annotations (line 141): numeric nuScenes annotations (event timestamps, frame identifiers, object categories, status labels) are manually verbalized; examples: spatial coordinate changes -> "lateral movements to the right"; object motion -> "moving away from the ego vehicle". Table 10 (line 417) shows a full example with Scene Token, Sequence ID, front/back descriptions in the 3-part structure ([1] Description of the Scene / [2] Key Changes Over Time / [3] Important Objects and Events from the Driver's Perspective), a Ground Truth Annotation line, and Start/End Index fields.
- QA generation prompt (Table 8, line 409): generates 10 Q&A pairs per call in "Q: ... A: ..." format, uses \{start_index\}, \{end_index\}, \{front_description\}, \{back_description\}, \{gt_description\}; includes the left/right mirror instruction for the back view: "objects on the left correspond to the ego vehicle's right, and objects on the right correspond to its left, based on the LiDAR perspective."
- LiDAR object categories allowed in descriptions (Table 7, line 405): "{Animal, pedestrian, stroller, wheelchair, barrier, debris, trafficcone, construction, motorcycle, bicycle, car, bus, trailer, truck, suv}"; prompt forbids "color, text, lighting, weather, or other 2D-specific details".
- Waymo variant (line 235): five front-facing cameras (front, front-left, front-right, side-left, side-right); "we replaced the front and rear views used in the nuScenes setup with left and right views during data generation."

## 6. Appendix hyperparameters

- **None present.** Appendix C = Metatoken formulation (line 375); Appendix E = scaling (Table 6, line 387); Appendix F = prompts (Tables 7–10, lines 405–417); Appendix G = qualitative figures. No learning rate, batch size, epoch count, optimizer, or LoRA configuration (rank/alpha/target modules) is stated anywhere in the paper. The only quantitative training facts are: single linear layer aligner, LoRA in stage 2, Vicuna-7b-v1.5 frozen, CLIP ViT-L/14 image encoder, cross-entropy loss, single RTX 4090, under 24 hours.

---

## Audit-relevant gaps and ambiguities summary

1. No mIoU formula (only "overlap between predicted and ground-truth temporal segments", line 129); segment representation (frame indices) must be inferred.
2. No points-per-frame count, voxelization parameters, or k (number of local features) specified; only CLS tokens are used for the LLM input (line 183).
3. No positional/temporal encoding over the T frames — plain concatenation of CLS embeddings.
4. No LR/batch/epochs/optimizer/LoRA-rank; "complete within 24 hours" on one RTX 4090 (line 211) is the only budget constraint.
5. Frame-number format is inconsistent across paper artifacts (see Section 2 above): the generation prompt mandates "from frame 000 to frame 000" (3-digit zero-padded), but examples/figures show "frame 10 to frame 22", "frame 0 to frame 7", "Frame 015", etc. Code must decide one canonical format; Table 8 (line 409) and Table 9's "from frame 004 to frame 006" (line 413) support zero-padded 3-digit.
6. Sequence length contradiction: "3 to 10 frames" (line 155) vs Table 2 "Min 5 / Max 19" (line 151); train count "14.8k" (line 157) vs Table 2 "148,271" (line 151).
7. Metatoken descriptions cover "the first and last frames referenced in the QA pair" (line 375), phrased in Sec 4.1 as "only for the initial and final frames" (line 187) — the QA-referenced reading (Appendix C) is the more precise one.
8. No `<video>` token or per-frame visual placeholder token is defined anywhere; the only special tokens are `<4DLiDAR>` and `<meta>`.

