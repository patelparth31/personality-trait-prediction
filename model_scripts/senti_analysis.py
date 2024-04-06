from transformers import pipeline
import matplotlib.pyplot as plt

def sentiment_analysis(text):
    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    classification = pipeline("sentiment-analysis", model=model_id)

    max_chunk_len = 256
    chunks = [text[i:i + max_chunk_len] for i in range(0, len(text), max_chunk_len)]
    chunk_sentiments = []
    for chunk in chunks:
        sentiment = classification(chunk)
        chunk_sentiments.append(sentiment)

    positive_count = sum(1 for sentiment in chunk_sentiments if sentiment[0]['label'] == 'positive')
    negative_count = sum(1 for sentiment in chunk_sentiments if sentiment[0]['label'] == 'negative')
    overall_sentiment = "POSITIVE" if positive_count > negative_count else "NEGATIVE"

    total_sentiments = len(chunk_sentiments)
    positive_percentage = (positive_count / total_sentiments) * 100
    negative_percentage = (negative_count / total_sentiments) * 100

    if positive_count == 0:
        positive_percentage = 100 - negative_percentage
    elif negative_count == 0:
        negative_percentage = 100 - positive_percentage

    # Saving a Graph
    # Data for the pie chart
    labels = ['Positive', 'Negative']
    sizes = [positive_percentage, negative_percentage]
    colors = ['#436850', '#B2533E']

    filepath = "top_10_words_frequency.png"
    
    # Plotting the pie chart
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', wedgeprops={"linewidth": 1, "edgecolor": "white"})
    plt.title('Impact of the Speech')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(filepath, dpi=300)


    return [overall_sentiment, positive_percentage, negative_percentage, filepath]
