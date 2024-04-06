from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def text_summarizer(text):
    try:
        model_name = 'google/pegasus-xsum'
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

        # Explicitly initialize weights
        model.model.decoder.embed_positions.weight.data.uniform_(-0.1, 0.1)
        model.model.encoder.embed_positions.weight.data.uniform_(-0.1, 0.1)

        max_chunk_len = 512
        chunks = [text[i:i + max_chunk_len] for i in range(0, len(text), max_chunk_len)]

        summaries = []
        for chunk in chunks:
            inputs = tokenizer(chunk, max_length=1024, return_tensors='pt', truncation=True)
            summary_ids = model.generate(inputs['input_ids'], max_length=200, num_beams=4, length_penalty=2.0,
                                         early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        final_summary = " ".join(summaries)

        return [summaries]
    except Exception as e:
        print("Error occurred:", e)
        return None


# def text_summarizer(text):
#     model_name = 'google/pegasus-xsum'
#     tokenizer = PegasusTokenizer.from_pretrained(model_name)
#     model = PegasusForConditionalGeneration.from_pretrained(model_name)
#
#     # Initialize the weights explicitly
#     model.model.decoder.embed_positions.weight.data.uniform_(-0.1, 0.1)
#     model.model.encoder.embed_positions.weight.data.uniform_(-0.1, 0.1)
#
#     max_chunk_len = 512
#     chunks = [text[i:i + max_chunk_len] for i in range(0, len(text), max_chunk_len)]
#
#     summaries = []
#     for chunk in chunks:
#         inputs = tokenizer(chunk, max_length=1024, return_tensors='pt', truncation=True)
#         summary_ids = model.generate(inputs['input_ids'], max_length=200, num_beams=4, length_penalty=2.0,
#                                      early_stopping=True)
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         summaries.append(summary)
#
#     final_summary = " ".join(summaries)
#
#     # Saving a plot ======================
#     # word_cloud = WordCloud(collocations=False, background_color='black').generate(text)
#     # word_freq = word_cloud.words_
#     # sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
#     # top_10_words = dict(sorted_word_freq[:10])
#     # actual_freq = [word_freq[word] * len(text.split()) for word in top_10_words]
#
#     # word_path = "../images/word_cloud.png"
#     # word_path = "../static/word_cloud.png"
#     # hist_path = "histogram.png"
#
#     # Plotting bar graph
#     # plt.figure(figsize=(10, 6))
#     # bars = plt.bar(top_10_words.keys(), actual_freq, color='skyblue')
#     # plt.xlabel('Words')
#     # plt.ylabel('Frequency')
#     # plt.title('Top 10 Words Frequency')
#     # plt.xticks(rotation=45)
#     # plt.tight_layout()
#     # plt.savefig(hist_path, dpi=300)
#     # plt.show()
#
#     # plt.imshow(word_cloud, interpolation='bilinear')
#     # plt.axis("off")
#     # plt.savefig(word_cloud, dpi=300)
#     # plt.show()
#
#     return [summaries]
