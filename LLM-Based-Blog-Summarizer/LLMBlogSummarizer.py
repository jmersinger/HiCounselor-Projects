# Before running, input the following to the command prompt:
# source venv/bin/activate
# pip install -r requirements.txt

# Also make sure you have added a blog/text file to be summarized.
# blog.txt is already in the directory, so if you add a different text file to be summarize, change the file path on line 14.

# To run:
# python3 LLM-Based-Blog-Summarizer/LLMBlogSummarizer.py


from transformers import BartTokenizer, TFBartForConditionalGeneration, T5Tokenizer, TFT5ForConditionalGeneration, PegasusTokenizer, TFPegasusForConditionalGeneration

file_path = open("LLM-Based-Blog-Summarizer/blog.txt", "r")
blog_content = file_path.read()


text = blog_content.lower()
cleaned_text = text.replace("\n\n", "")

#---Summary Using Bart Tokenizer---

# Load the tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = TFBartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Tokenize the input text
encoded_text = tokenizer.encode(cleaned_text, return_tensors="tf", max_length=1024, truncation=True)

# Generate the summary
summary = model.generate(encoded_text, max_length=300, num_beams=4, early_stopping=True)

# Decode and print the generated summary
summary_bart = tokenizer.decode(summary[0], skip_special_tokens=True)
print(summary_bart)

#Export the Generated text ("summary_bart.txt")
output_filepath = "LLM-Based-Blog-Summarizer/summary_bart.txt"
with open(output_filepath, "w", encoding="utf-8") as output_file:
    output_file.write(summary_bart)

#---Summary Using T5 Tokenizer---

# Load the tokenizer and model
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = TFT5ForConditionalGeneration.from_pretrained('t5-base')

# Tokenize the input text
t5_encoded_text = t5_tokenizer.encode("summarize:" + cleaned_text, return_tensors="tf", max_length=1024, truncation=True)

# Generate the summary
t5_summary = t5_model.generate(t5_encoded_text, max_length=300, num_beams=4, early_stopping=True)

# Decode and print the generated summary
summary_t5 = t5_tokenizer.decode(t5_summary[0], skip_special_tokens=True)

#Export the Generated text ("summary_t5.txt")
output_filepath = "LLM-Based-Blog-Summarizer/summary_t5.txt"
with open(output_filepath, "w", encoding="utf-8") as output_file:
    output_file.write(summary_t5)
    
#---Summary with Pegasus Tokenizer---

# Load the tokenizer and model
pegasus_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
pegasus_model = TFPegasusForConditionalGeneration.from_pretrained('google/pegasus-large')

# Tokenize the input text
tokenized_text = pegasus_tokenizer.encode(cleaned_text, return_tensors="tf", max_length=1024, truncation=True)

# Generate the summary
pegasus_summary = pegasus_model.generate(tokenized_text, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)

# Decode and print the generated summary
summary_pegasus = pegasus_tokenizer.decode(pegasus_summary[0], skip_special_tokens=True)
print(summary_pegasus)

# Export the Generated text ("summary_pegasus.txt")
output_filepath = "LLM-Based-Blog-Summarizer/summary_pegasus.txt"
with open(output_filepath, "w", encoding="utf-8") as output_file:
     output_file.write(summary_pegasus)