<div align="center">

<h2><a href="https://arxiv.org/abs/2601.23224">[ICML 2026] Video-o3: Native Interleaved Clue Seeking for Long Video Multi-Hop Reasoning</a></h2>

[Xiangyu Zeng](https://lanxingxuan.github.io/)\*, Zhiqiu Zhang\*, [Yuhan Zhu](https://zyuhan1999.github.io/)\*, [Xinhao Li](https://leexinhao.github.io/)\*, Zikang Wang\*, Changlian Ma, [Qingyu Zhang](https://happyzqy.github.io/), Zizheng Huang, Kun Ouyang, Tianxiang Jiang, Ziang Yan, Yi Wang, Hongjie Zhang, Yali Wang, and [Limin Wang](https://wanglimin.github.io/)†

<br>

<p align="center">
  <a href="https://mcg-nju.github.io/Video-o3/">
    <img src="https://img.shields.io/badge/Project-Page-orange.svg" alt="Homepage">
  </a>
  <a href="https://arxiv.org/abs/2601.23224">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper">
  </a>
  <a href="https://huggingface.co/collections/MCG-NJU/video-o3">
    <img src="https://img.shields.io/badge/Model-HuggingFace-yellow.svg" alt="Model">
  </a>
  <a href="https://huggingface.co/datasets/MCG-NJU/Seeker-173K">
    <img src="https://img.shields.io/badge/Data-Seeker--173K-blue.svg" alt="Data">
  </a>
</p>

</div>

## :fire: Updates
- [x] **2026/06/14**: 🔥🔥🔥Release the training and evaluation code for the Qwen3-VL version. See the [qwen3vl](https://github.com/MCG-NJU/Video-o3/tree/qwen3vl) branch.
- [x] **2026/02/16**: 🔥🔥🔥Release the training code for SFT and RL.
- [x] **2026/02/15**: 🔥🔥🔥Release the evaluation code for Video-o3.
- [x] **2026/02/10**: 🔥🔥🔥Release the checkpoint of **[Video-o3 (RL)](https://huggingface.co/collections/MCG-NJU/video-o3)** and **[Video-o3 (SFT+RL)](https://huggingface.co/collections/MCG-NJU/video-o3)**.
- [x] **2026/02/10**: 🔥🔥🔥Release **[Seeker-173K](https://huggingface.co/datasets/MCG-NJU/Seeker-173K)**, a large-scale dataset comprising 173K high-quality tool-interaction trajectories for effective supervised and reinforcement learning.
- [x] **2026/01/30**: 🔥🔥🔥Release the paper of **[Video-o3](https://arxiv.org/abs/2601.23224)**, a novel framework that supports native interleaved clue seeking for long video multi-hop reasoning.

## 📑 Todo List
- [ ] Refine the user documentation and tutorials.
- [ ] Update with more engaging and interactive demos.
- [ ] Provide a streamlined guide for quick inference.



## 🧠 Motivation: Thinking with Videos

Current Multimodal Large Language Models (MLLMs) struggle with long videos because they typically rely on uniform frame sampling and single-turn inference. This approach often dilutes critical visual evidence within redundant background content.

Video-o3 introduces a paradigm shift by mimicking human behavior. Instead of watching a video passively, it actively explores the content. The model iteratively discovers salient visual clues, inspects key segments with fine-grained detail, and adaptively terminates the search once sufficient evidence is acquired.

<div align="center">
  <img src="src/Figure1.png" width="100%"/>
  <br>
  <em>Overview of Video-o3. Guided by the user query, the model actively identifies and localizes critical visual clues using native interleaved tool invocation. It autonomously decides whether to continue searching or to conclude the reasoning process.</em>
</div>

**Key Features:**

*   **Goal-Driven Exploration:** Unlike models that scan the whole video coarsely, Video-o3 starts with a coarse scan and iteratively focuses on informative segments.
*   **Native Interleaved Tool Use:** The model supports "clue seeking" and "answer reasoning" within a single shared context, rather than decoupled modules.

## 🛠️ Method: Native Interleaved Tool Invocation

Video-o3 is designed to solve the challenges of attention dispersion and contextual efficiency in long-video processing. The framework operates on a Think-and-Tool cycle. The model generates structured directives containing temporal windows and visual token quotas. It dynamically invokes the VideoCrop tool to inspect target segments with adaptive spatiotemporal resolution.

<div align="center">
  <img src="src/Figure3.png" width="100%"/>
  <br>
  <em>Architectural Overview. Video-o3 dynamically executes tool invocations based on previous reasoning to scrutinize specific clue segments. The Vision Encoder uses adaptive flexible sampling, while the LLM Decoder manages the interleaved "Think," "Tool," and "Answer" tokens.</em>
</div>

**Core Technical Innovations:**

*   **Task-Decoupled Attention Masking:** To prevent attention dispersion, TDAM isolates per-step concentration. During clue seeking, the model attends only to the global context; during answering, it focuses on high-resolution tool observations.
*   **Verifiable Trajectory-Guided Reward:** To control context length and cost, we introduce a reward mechanism that balances exploration coverage with reasoning efficiency, encouraging the model to terminate precisely when evidence is sufficient.

## 📊 Data Construction: Pipeline and Seeker-173K

Training a model to perform native interleaved tool invocation requires high-quality exploration trajectories, which are scarce in existing datasets. To address this, we developed a scalable automated data synthesis pipeline.

<div align="center">
  <img src="src/Figure5.png" width="100%"/>
  <br>
  <em>The Data Construction Pipeline. We transform "Video-Question-Answer" triplets into explicit tool exploration trajectories via a four-stage process: Clue Localization, Validity Verification, Trajectory Generation, and Logical Consistency Checks.</em>
</div>

**About Seeker-173K:**

*   **Structure:** The dataset is stratified into a four-quadrant taxonomy based on evidence cardinality and visual saliency, covering tasks from "Single-Clue Direct Answering" to complex "Multi-Clue Tool Invocation".
*   **Quality:** Human verification is enforced through random sampling in all stages. The pipeline rigorously filters out flawed instances, preserving only those with sound logic and factual visual evidence.

## 📈 Performance

<table border="1" style="font-size: 0.9em; width: 100%; text-align: center; border-collapse: collapse;">
  <thead>
    <tr>
      <th rowspan="2" style="vertical-align: middle;">Methods</th>
      <th rowspan="2" style="vertical-align: middle;">Sizes</th>
      <th>VideoMME</th>
      <th>MLVU</th>
      <th>LVBench</th>
      <th>LongVideoBench</th>
      <th>VideoMMMU</th>
      <th>MMVU</th>
      <th>Video-Holmes</th>
    </tr>
    <tr>
      <th>Avg</th>
      <th>M-Avg</th>
      <th>Avg</th>
      <th>Avg</th>
      <th>Overall</th>
      <th>M-Avg</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="9" style="text-align: left;"><i>Open-source Single-Turn Video MLLMs</i></td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen2.5-VL</td>
      <td>7B</td>
      <td>65.1</td>
      <td>70.2</td>
      <td>45.3</td>
      <td>56.0</td>
      <td>47.4</td>
      <td>61.3</td>
      <td>34.7</td>
    </tr>
    <tr>
      <td style="text-align: left;">LLaVA-Video</td>
      <td>7B</td>
      <td>63.3</td>
      <td>70.8</td>
      <td>-</td>
      <td>58.2</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="text-align: left;">Video-R1</td>
      <td>7B</td>
      <td>61.4</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>52.4</td>
      <td>63.8</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="text-align: left;">Rewatch-R1</td>
      <td>7B</td>
      <td>65.6</td>
      <td>-</td>
      <td>43.3</td>
      <td>-</td>
      <td>51.9</td>
      <td>-</td>
      <td>44.3</td>
    </tr>
    <tr>
      <td style="text-align: left;">Video-Thinker</td>
      <td>7B</td>
      <td>-</td>
      <td>-</td>
      <td>37.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>43.2</td>
    </tr>
    <tr>
      <td colspan="9" style="text-align: left;"><i>Open-source Decoupled Iterative Reasoning Video MLLMs</i></td>
    </tr>
    <tr>
      <td style="text-align: left;">Video-RTS</td>
      <td>7B</td>
      <td>63.0</td>
      <td>-</td>
      <td>-</td>
      <td>56.6</td>
      <td>52.7</td>
      <td>66.4</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="text-align: left;">Video-MTR</td>
      <td>7B</td>
      <td>59.0</td>
      <td>59.7</td>
      <td>38.6</td>
      <td>56.4</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="text-align: left;">LOVE-R1</td>
      <td>7B</td>
      <td>66.2</td>
      <td>67.4</td>
      <td>48.2</td>
      <td>60.1</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td colspan="9" style="text-align: left;"><i>Open-source Native Multi-turn Tool Invocation Video MLLMs</i></td>
    </tr>
    <tr>
      <td style="text-align: left;">Conan</td>
      <td>7B</td>
      <td>60.5</td>
      <td>63.4</td>
      <td>39.2</td>
      <td>56.6</td>
      <td>-</td>
      <td>-</td>
      <td>44.6</td>
    </tr>
    <tr>
      <td style="text-align: left;">LongVT</td>
      <td>7B</td>
      <td>64.3</td>
      <td>-</td>
      <td>41.3</td>
      <td>-</td>
      <td>45.4</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="text-align: left;">Video-Zoomer</td>
      <td>7B</td>
      <td>65.2</td>
      <td>-</td>
      <td>41.5</td>
      <td>57.7</td>
      <td>52.2</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="text-align: left;"><b>Video-o3 (RL)</b></td>
      <td><b>7B</b></td>
      <td><b>66.1</b></td>
      <td><b>71.9</b></td>
      <td><b>47.5</b></td>
      <td><b>59.3</b></td>
      <td><b>50.0</b></td>
      <td><b>66.9</b></td>
      <td><b>46.1</b></td>
    </tr>
    <tr>
      <td style="text-align: left;"><b>Video-o3 (SFT+RL)</b></td>
      <td><b>7B</b></td>
      <td><b>66.5</b></td>
      <td><b>72.1</b></td>
      <td><b>47.6</b></td>
      <td><b>60.5</b></td>
      <td><b>51.7</b></td>
      <td><b>67.2</b></td>
      <td><b>46.5</b></td>
    </tr>
  </tbody>
</table>

**Comparison of our method with existing approaches on video question answering tasks across various benchmarks.** Video-o3 significantly outperforms previous methods in long video understanding benchmarks while also demonstrating strong performance in multiple video inference benchmarks.

## 🖊️ Citation

```bibtex
@article{zeng2026video,
  title={Video-o3: Native Interleaved Clue Seeking for Long Video Multi-Hop Reasoning},
  author={Zeng, Xiangyu and Zhang, Zhiqiu and Zhu, Yuhan and Li, Xinhao and Wang, Zikang and Ma, Changlian and Zhang, Qingyu and Huang, Zizheng and Ouyang, Kun and Jiang, Tianxiang and others},
  journal={arXiv preprint arXiv:2601.23224},
  year={2026}
}
```


## :dizzy: Acknowledgement

Thanks to the open source of the following projects:
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- [Mini-o3](https://arxiv.org/abs/2509.07969)
- [LongVT](https://arxiv.org/abs/2511.20785)
