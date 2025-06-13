# Finetuned-Medical-Multimodal-RAG

Requirements:
1. GPT-2 Fine-Tuned on MedQuAD
This is a domain-adapted GPT-2 model fine-tuned using the MedQuAD dataset for medical question-answering.
🔗 Download from Hugging Face: https://huggingface.co/sarnsrun/gpt2-medquad-finetuned

2. BLIP-VQA Fine-Tuned on VQA-RAD
This is a BLIP model fine-tuned on the VQA-RAD dataset for visual question answering in the medical domain.
🔗 Download from Hugging Face: https://huggingface.co/sarnsrun/blip-vqa-finetuned

3. VQA-RAD Dataset
A processed version of the VQA-RAD dataset, containing only open-ended questions.
🔗 Download: https://huggingface.co/datasets/sarnsrun/vqa-rad

4. ROCO Dataset
Obtained from: https://github.com/razorx89/roco-dataset/tree/master/data

Citations:
Ben Abacha, A., & Demner-Fushman, D. (2019). A question-entailment approach to question answering. BMC Bioinformatics, 20(1), 511. https://doi.org/10.1186/s12859-019-3119-4

Lau, J. J., Gayen, S., Demner-Fushman, D., & Ben Abacha, A. (2018). VQA-RAD: A dataset of clinically generated visual questions and answers about radiology images. OSF. https://doi.org/10.17605/OSF.IO/89KPS

O. Pelka, S. Koitka, J. Rückert, F. Nensa, C.M. Friedrich,
"Radiology Objects in COntext (ROCO): A Multimodal Image Dataset".
MICCAI Workshop on Large-scale Annotation of Biomedical Data and Expert Label Synthesis (LABELS) 2018, September 16, 2018, Granada, Spain. Lecture Notes on Computer Science (LNCS), vol. 11043, pp. 180-189, Springer Cham, 2018.
doi: 10.1007/978-3-030-01364-6_20
