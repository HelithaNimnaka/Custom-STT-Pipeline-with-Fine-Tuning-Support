# 🎙️ Custom STT Pipeline with Fine-Tuning Support

## 📌 Model:
**Whisper Tiny** ([`openai/whisper-tiny`](https://github.com/openai/whisper))

## 📚 Fine-tuning Dataset:
[Hugging Face Common Voice (100 samples demo)](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)

---

## 🚀 How to Use:

### 1️⃣ Clone the Repository
```
git clone https://github.com/HelithaNimnaka/Custom-STT-Pipeline-with-Fine-Tuning-Support.git
cd Custom-STT-Pipeline-with-Fine-Tuning-Support
```
### 2️⃣ Install Requirements
```
pip install -r requirements.txt
```
### 3️⃣ Run the App
```
python app.py
```
👉 You will see an interface like this:  
![Streamlit Interface Example](streamlit_example.png)


---

## 🛠 Fine-Tuning Steps:

1. **Add Hugging Face Token:**  
   In [`finetuning.py`](finetuning.py):
   ```python
   token = "add_your_huggingface_token_here"
2. **Run Fine-Tuning Script:**<br>
   Customize the dataset/model if needed and run:
   ```
   python finetuning.py
   ```
3. **Update Weights:**<br>
   Load the fine-tuned model weights into [`stt_pipeline.py`](stt_pipeline.py) for inference.
   
---

### ✅ Done! You now have a fully functional Speech-to-Text pipeline with fine-tuning support.

---

## 📈 Scaling to Large Datasets

When you move beyond a 100-sample demo to tens or hundreds of thousands of hours of audio, consider:

1. **Streaming & Sharding**  
   - Use `datasets.load_dataset(..., streaming=True)` to avoid full downloads.  
   - Store your data in sharded archives (e.g. WebDataset `.tar` shards) on S3 or Google Cloud Storage (GCS) and stream directly.

2. **Distributed Training**  
   - Leverage Hugging Face Accelerate or DeepSpeed for multi-GPU/multi-node training.  
   - Use gradient accumulation to simulate larger batch sizes without out of memory.

3. **Efficient Preprocessing**  
   - Precompute and cache audio features (e.g. spectrograms) to avoid repeated FFTs.  
   - Use multiprocessing (`num_proc`) in `dataset.map()` to parallelize.

4. **Mixed Precision & Memory Optimizations**  
   - Enable `fp16=True` (or `bf16=True`) in `TrainingArguments` for half-precision.  
   - Use model offloading (DeepSpeed ZeRO) to fit larger models.

5. **Data Sampling & Curriculum**  
   - Start with shorter clips or higher-quality audio, then gradually include noisier or longer samples.  
   - Use bucketing or dynamic batching to group similar-length inputs.

6. **Monitoring & Checkpointing**  
   - Save checkpoints (`save_steps`) to S3/GCS regularly.  
   - Use WandB or TensorBoard to track loss and throughput.

7. **Cloud & Auto-Scaling**  
   - Spin up spot GPU instances (AWS, GCP, Azure) with auto-recovery on preemption.  
   - Automate job submission with Slurm/KubeFlow for large-scale experiments.


With these strategies, you can fine-tune Whisper on **hundreds of thousands of hours** of real-world audio while keeping compute, I/O, and costs under control. 🚀


---

## 🔧 Parameter-Efficient Fine-Tuning (PEFT)

When you want to fine-tune large models with limited compute or storage, consider PEFT methods:

1. **LoRA (Low-Rank Adaptation)**  
   - Injects small low-rank update matrices into the attention and feed-forward layers.  
   - Only these adapter matrices are trained (≈0.1–1% of model parameters).  
   - Example with the `peft` library:
     ```python
     from peft import LoraConfig, get_peft_model

     # Configure LoRA
     lora_config = LoraConfig(
         r=16, 
         lora_alpha=32, 
         target_modules=["q_proj", "v_proj"]
     )
     peft_model = get_peft_model(model, lora_config)
     ```

2. **Adapter Modules**  
   - Insert small “bottleneck” adapter layers between each transformer block.  
   - Only adapter weights are updated; base model stays frozen.

3. **Prefix Tuning**  
   - Prepends a learnable continuous “prefix” of tokens to every layer’s key/value inputs.  
   - Trains only the prefix embeddings, leaving the main model untouched.

4. **Prompt Tuning**  
   - Learns a set of soft prompt embeddings that sit before the input tokens.  
   - Extreme parameter efficiency: only prompt embeddings are trained.

By using PEFT methods like LoRA or adapters, you can adapt Whisper (or any large model) with minimal additional parameters, dramatically reducing GPU memory and storage requirements while still achieving strong performance.  


