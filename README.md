# Enhancing LLM-Based Text-to-SQL Evaluation: Novel Metrics and Prompting Strategies

Text-to-SQL is one of the most prominent areas where Large Language Models (LLMs)
have found extensive application. While current Text-to-SQL systems leveraging
LLMs and pre-trained models are typically evaluated using standard metrics such
as Exact Match (EM), Execution Accuracy (EX), and Valid Efficiency Score (VES),
these metrics overlook two critical aspects: the consistency of model outputs and
the handling of duplicate rows in query results.

In this work, we introduce two additional evaluation metrics to address these gaps.
First, a Consistency Metric, which evaluates a model’s reliability in generating iden-
tical SQL queries for repeated inputs. Second, a Symmetric Difference Metric(SDM),
which accounts for duplicate rows in query results, an issue otherwise ignored by
the existing Execution Accuracy (EX) metric.

To enhance model comprehension, we experimented with verbal schema descriptions
in prompts and latest database schema representaions.

To tackle complex questions involving sub-queries and intricate joins, we introduced
a novel variant of Chain-of-Thought (CoT) prompting, called Series of Questions
(SoQ), which breaks down complex queries into simpler, sequential questions,
guiding the model toward more accurate SQL generation.

Our findings suggest that combining Consistency and Symmetric Difference metrics
with existing evaluation methods provides a more holistic framework for assessing
the Text-to-SQL systems. Furthermore, our exploration of schema augmentation and
the novel Series of Questions (SoQ) prompting strategy demonstrates how prompt
engineering can enhance query generation. Together, these contributions provide a
more comprehensive approach to evaluating Text-to-SQL systems.

---

 
## Publication:
### Comparing Accuracy and Consistency: LLMs vs. SOTA Deep Learning Models in Text-to-SQL

Presented at the 1st Workshop on AI for Scientific and General Purpose Computing (AI-SPC) held in conjunction with the 31st IEEE International Conference on High-Performance Computing, Data, & Analytics (HiPC 2024)

🔗 Published at HiPCW 2024: https://ieeexplore.ieee.org/document/10898697
