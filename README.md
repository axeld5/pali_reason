# Pali-Reason Project  

## Overview  
This project utilizes **Gemini 2.0 Flash Thiking** to generate distilled thoughts from the **MathVista Dataset**, enabling sophisticated reasoning capabilities to finetune **PaliGemma2-3B-448pt**.

---

## Project Structure  
```
.
├── training_data/                      # Contains training datasets.
├── outputs/                            # Contains results from the inference files.
├── generating_training_examples.ipynb  # Notebook for generating initial thoughts.
├── formatting_training_dataset.ipynb   # Notebook for getting the right data for training.
├── finetuning.py                       # Script for fine-tuning the model (requires 1 H100 GPU).
├── inference.py                        # Script for inference with the base model.
├── inference_finetuned.py              # Script for inference with the fine-tuned model.
├── visualising_examples.ipynb          # Script to visualise the outputs of the different models.
├── answer_eval.ipynb                   # Notebook for evaluating model-generated answers.
├── requirements.txt                    # Required Python dependencies.
├── .env                                # API keys for Anthropic and Google.
└── LICENSE                             # MIT License.
```

---

## Setup Instructions  

### 1. Install Dependencies  
Ensure Python is installed, then run:  
```bash  
pip install -r requirements.txt  
```  

### 2. Environment Variables  
Create a `.env` file in the project directory with the following keys:  
```env  
ANTHROPIC_API_KEY=<your-anthropic-api-key>  
GOOGLE_API_KEY=<your-google-api-key>  
```  

---

## Usage  

### 1. Thought Generation  
Run `generating_training_examples.ipynb` to generate initial thoughts from the MathVista dataset.  

### 2. Data Correction and Formatting  
- Use `formatting_training_dataset` to correct the generated thoughts and format the corrected data with `from_csv_to_train.ipynb` to prepare it for training.  

### 3. Fine-Tuning  
Fine-tune the model using the `finetuning.py` script. Ensure access to an H100 GPU for efficient processing:  
```bash  
python finetuning.py  
```  

### 4. Inference  
- For base model inference:  
  ```bash  
  python inference.py  
  ```  
- For fine-tuned model inference:  
  ```bash  
  python inference_finetuned.py  
  ```  

### 5. Evaluation  
Evaluate the results using the `answer_eval.ipynb` notebook.  

---

## License  
This project is licensed under the [MIT License](LICENSE).  

## Contributions  
Contributions are welcome! Please create a pull request or raise an issue if you'd like to improve the project.  

---  

For any questions or feedback, please contact the project maintainers.

