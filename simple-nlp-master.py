"""
6_simple_nlp_master.py
Comprehensive Simple NLP Pipeline with Traditional Methods
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import gensim
from gensim import corpora, models
from gensim.models import Word2Vec, FastText, LdaModel
from gensim.models.phrases import Phrases, Phraser
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except:
    print("NLTK downloads completed or failed - continuing...")

class SimpleTextPreprocessor:
    """Comprehensive text preprocessing pipeline"""
    
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.has_spacy = True
        except:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.has_spacy = False
            self.nlp = None
    
    def clean_text(self, text, 
                  remove_punct=True, 
                  remove_numbers=True, 
                  remove_stopwords=True,
                  lowercase=True,
                  remove_extra_spaces=True):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        if lowercase:
            text = text.lower()
        
        if remove_punct:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        if remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
        
        if remove_stopwords:
            tokens = text.split()
            tokens = [token for token in tokens if token not in self.stop_words]
            text = ' '.join(tokens)
        
        return text
    
    def tokenize_text(self, text, method='word'):
        """Tokenize text using different methods"""
        if method == 'word':
            return word_tokenize(text)
        elif method == 'sentence':
            return sent_tokenize(text)
        elif method == 'regex':
            return self.tokenizer.tokenize(text)
        elif method == 'spacy' and self.has_spacy:
            doc = self.nlp(text)
            return [token.text for token in doc]
        else:
            return text.split()
    
    def stem_text(self, text):
        """Stem text using Porter Stemmer"""
        tokens = self.tokenize_text(text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)
    
    def lemmatize_text(self, text):
        """Lemmatize text using WordNet"""
        tokens = self.tokenize_text(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    
    def remove_special_characters(self, text, keep_alpha_numeric=True):
        """Remove special characters"""
        if keep_alpha_numeric:
            return re.sub(r'[^a-zA-Z0-9\s]', '', text)
        else:
            return re.sub(r'[^\w\s]', '', text)
    
    def preprocess_pipeline(self, texts, steps=None):
        """Complete preprocessing pipeline"""
        if steps is None:
            steps = ['clean', 'lemmatize']
        
        processed_texts = []
        
        for text in texts:
            processed_text = str(text)
            
            if 'clean' in steps:
                processed_text = self.clean_text(processed_text)
            
            if 'stem' in steps:
                processed_text = self.stem_text(processed_text)
            
            if 'lemmatize' in steps:
                processed_text = self.lemmatize_text(processed_text)
            
            if 'remove_special' in steps:
                processed_text = self.remove_special_characters(processed_text)
            
            processed_texts.append(processed_text)
        
        return processed_texts

class TextAnalyzer:
    """Comprehensive text analysis toolkit"""
    
    def __init__(self):
        self.preprocessor = SimpleTextPreprocessor()
        self.sia = SentimentIntensityAnalyzer()
    
    def basic_statistics(self, texts):
        """Compute basic text statistics"""
        stats = {
            'num_documents': len(texts),
            'total_words': 0,
            'total_characters': 0,
            'avg_words_per_doc': 0,
            'avg_chars_per_doc': 0,
            'vocabulary_size': 0
        }
        
        all_words = []
        for text in texts:
            words = self.preprocessor.tokenize_text(text)
            all_words.extend(words)
            stats['total_words'] += len(words)
            stats['total_characters'] += len(text)
        
        stats['avg_words_per_doc'] = stats['total_words'] / len(texts)
        stats['avg_chars_per_doc'] = stats['total_characters'] / len(texts)
        stats['vocabulary_size'] = len(set(all_words))
        
        return stats
    
    def word_frequency(self, texts, top_n=20):
        """Compute word frequency distribution"""
        all_words = []
        for text in texts:
            cleaned_text = self.preprocessor.clean_text(text)
            words = self.preprocessor.tokenize_text(cleaned_text)
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        return word_freq.most_common(top_n)
    
    def ngram_analysis(self, texts, n=2, top_n=10):
        """Compute n-gram frequencies"""
        all_ngrams = []
        for text in texts:
            cleaned_text = self.preprocessor.clean_text(text)
            tokens = self.preprocessor.tokenize_text(cleaned_text)
            text_ngrams = list(ngrams(tokens, n))
            all_ngrams.extend(text_ngrams)
        
        ngram_freq = Counter(all_ngrams)
        return ngram_freq.most_common(top_n)
    
    def sentiment_analysis(self, texts):
        """Perform sentiment analysis using multiple methods"""
        sentiments = []
        
        for text in texts:
            # VADER sentiment
            vader_scores = self.sia.polarity_scores(text)
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            sentiment = {
                'text': text,
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu'],
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity,
                'sentiment_label': self._get_sentiment_label(vader_scores['compound'])
            }
            
            sentiments.append(sentiment)
        
        return pd.DataFrame(sentiments)
    
    def _get_sentiment_label(self, compound_score):
        """Convert compound score to label"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def part_of_speech_analysis(self, texts):
        """Perform POS tagging analysis"""
        pos_counts = Counter()
        
        for text in texts:
            tokens = self.preprocessor.tokenize_text(text)
            pos_tags = pos_tag(tokens)
            pos_counts.update([tag for word, tag in pos_tags])
        
        return pos_counts
    
    def named_entity_recognition(self, texts):
        """Basic named entity recognition"""
        entities = []
        
        for text in texts:
            # Using NLTK
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            text_entities = []
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_text = ' '.join(c[0] for c in chunk)
                    entity_type = chunk.label()
                    text_entities.append((entity_text, entity_type))
            
            entities.append({
                'text': text,
                'entities': text_entities
            })
        
        return entities
    
    def create_word_cloud(self, texts, output_file='wordcloud.png'):
        """Create and display word cloud"""
        all_text = ' '.join(texts)
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100
        ).generate(all_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return wordcloud

class TextVectorizer:
    """Comprehensive text vectorization methods"""
    
    def __init__(self):
        self.vectorizers = {}
        self.models = {}
    
    def tfidf_vectorize(self, texts, max_features=5000, ngram_range=(1, 2), **kwargs):
        """TF-IDF vectorization"""
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            **kwargs
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        self.vectorizers['tfidf'] = vectorizer
        
        return tfidf_matrix, vectorizer
    
    def count_vectorize(self, texts, max_features=5000, ngram_range=(1, 2), **kwargs):
        """Count vectorization"""
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            **kwargs
        )
        
        count_matrix = vectorizer.fit_transform(texts)
        self.vectorizers['count'] = vectorizer
        
        return count_matrix, vectorizer
    
    def word2vec_embedding(self, texts, vector_size=100, window=5, min_count=2):
        """Word2Vec embeddings"""
        tokenized_texts = [text.split() for text in texts]
        
        model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        
        self.models['word2vec'] = model
        
        # Create document embeddings by averaging word vectors
        doc_embeddings = []
        for text in tokenized_texts:
            word_vectors = []
            for word in text:
                if word in model.wv:
                    word_vectors.append(model.wv[word])
            
            if word_vectors:
                doc_embedding = np.mean(word_vectors, axis=0)
            else:
                doc_embedding = np.zeros(vector_size)
            
            doc_embeddings.append(doc_embedding)
        
        return np.array(doc_embeddings), model
    
    def fasttext_embedding(self, texts, vector_size=100, window=5, min_count=2):
        """FastText embeddings"""
        tokenized_texts = [text.split() for text in texts]
        
        model = FastText(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        
        self.models['fasttext'] = model
        
        # Create document embeddings
        doc_embeddings = []
        for text in tokenized_texts:
            word_vectors = []
            for word in text:
                word_vectors.append(model.wv[word])
            
            doc_embedding = np.mean(word_vectors, axis=0)
            doc_embeddings.append(doc_embedding)
        
        return np.array(doc_embeddings), model

class TopicModeling:
    """Topic modeling using various algorithms"""
    
    def __init__(self):
        self.models = {}
    
    def lda_topic_modeling(self, texts, num_topics=5, **kwargs):
        """Latent Dirichlet Allocation"""
        # Preprocess and create dictionary
        tokenized_texts = [text.split() for text in texts]
        dictionary = corpora.Dictionary(tokenized_texts)
        
        # Create corpus
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            **kwargs
        )
        
        self.models['lda'] = lda_model
        self.dictionary = dictionary
        self.corpus = corpus
        
        return lda_model, dictionary, corpus
    
    def sklearn_lda(self, vectorized_texts, num_topics=5, **kwargs):
        """LDA using scikit-learn"""
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            **kwargs
        )
        
        lda_topics = lda.fit_transform(vectorized_texts)
        self.models['sklearn_lda'] = lda
        
        return lda_topics, lda
    
    def nmf_topic_modeling(self, vectorized_texts, num_topics=5, **kwargs):
        """Non-negative Matrix Factorization"""
        nmf = NMF(
            n_components=num_topics,
            random_state=42,
            **kwargs
        )
        
        nmf_topics = nmf.fit_transform(vectorized_texts)
        self.models['nmf'] = nmf
        
        return nmf_topics, nmf
    
    def get_topic_words(self, model, vectorizer, n_words=10):
        """Extract top words for each topic"""
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append((topic_idx, top_words))
        
        return topics
    
    def visualize_topics(self, model, vectorizer, n_words=10):
        """Visualize topics with top words"""
        topics = self.get_topic_words(model, vectorizer, n_words)
        
        fig, axes = plt.subplots(1, len(topics), figsize=(15, 5))
        if len(topics) == 1:
            axes = [axes]
        
        for idx, (topic_idx, words) in enumerate(topics):
            ax = axes[idx]
            y_pos = np.arange(len(words))
            ax.barh(y_pos, range(len(words), 0, -1))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.set_title(f'Topic {topic_idx + 1}')
            ax.invert_yaxis()
        
        plt.tight_layout()
        plt.show()

class TextClustering:
    """Text clustering using various methods"""
    
    def __init__(self):
        self.models = {}
    
    def kmeans_clustering(self, embeddings, n_clusters=5, **kwargs):
        """K-means clustering"""
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            **kwargs
        )
        
        clusters = kmeans.fit_predict(embeddings)
        self.models['kmeans'] = kmeans
        
        # Calculate silhouette score
        if len(set(clusters)) > 1:
            silhouette_avg = silhouette_score(embeddings, clusters)
        else:
            silhouette_avg = -1
        
        return clusters, kmeans, silhouette_avg
    
    def dbscan_clustering(self, embeddings, eps=0.5, min_samples=5, **kwargs):
        """DBSCAN clustering"""
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            **kwargs
        )
        
        clusters = dbscan.fit_predict(embeddings)
        self.models['dbscan'] = dbscan
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        return clusters, dbscan, n_clusters
    
    def visualize_clusters(self, embeddings, clusters, method='tsne'):
        """Visualize clusters using dimensionality reduction"""
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = TruncatedSVD(n_components=2)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=clusters, 
            cmap='viridis',
            alpha=0.7
        )
        plt.colorbar(scatter)
        plt.title(f'Text Clusters ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()
        
        return embeddings_2d

class SimpleNLPipeline:
    """Complete simple NLP pipeline"""
    
    def __init__(self):
        self.preprocessor = SimpleTextPreprocessor()
        self.analyzer = TextAnalyzer()
        self.vectorizer = TextVectorizer()
        self.topic_modeler = TopicModeling()
        self.clusterer = TextClustering()
        self.results = {}
    
    def run_complete_analysis(self, texts, analysis_types=None):
        """Run complete NLP analysis"""
        if analysis_types is None:
            analysis_types = ['preprocessing', 'statistics', 'sentiment', 'topics', 'clustering']
        
        print("Starting Complete NLP Analysis...")
        
        # Preprocessing
        if 'preprocessing' in analysis_types:
            print("1. Text Preprocessing...")
            processed_texts = self.preprocessor.preprocess_pipeline(texts)
            self.results['processed_texts'] = processed_texts
        else:
            processed_texts = texts
        
        # Basic Statistics
        if 'statistics' in analysis_types:
            print("2. Basic Statistics...")
            stats = self.analyzer.basic_statistics(processed_texts)
            self.results['statistics'] = stats
            print(f"Documents: {stats['num_documents']}, Vocabulary: {stats['vocabulary_size']}")
        
        # Word Frequency
        if 'frequency' in analysis_types:
            print("3. Word Frequency Analysis...")
            word_freq = self.analyzer.word_frequency(processed_texts)
            self.results['word_frequency'] = word_freq
        
        # Sentiment Analysis
        if 'sentiment' in analysis_types:
            print("4. Sentiment Analysis...")
            sentiment_df = self.analyzer.sentiment_analysis(texts)  # Use original for sentiment
            self.results['sentiment'] = sentiment_df
        
        # Vectorization
        if 'vectorization' in analysis_types:
            print("5. Text Vectorization...")
            tfidf_matrix, tfidf_vectorizer = self.vectorizer.tfidf_vectorize(processed_texts)
            self.results['tfidf_matrix'] = tfidf_matrix
            self.results['tfidf_vectorizer'] = tfidf_vectorizer
        
        # Topic Modeling
        if 'topics' in analysis_types:
            print("6. Topic Modeling...")
            if 'tfidf_matrix' in self.results:
                lda_topics, lda_model = self.topic_modeler.sklearn_lda(self.results['tfidf_matrix'])
                self.results['lda_topics'] = lda_topics
                self.results['lda_model'] = lda_model
                
                # Get topic words
                topic_words = self.topic_modeler.get_topic_words(
                    lda_model, 
                    self.results['tfidf_vectorizer']
                )
                self.results['topic_words'] = topic_words
        
        # Text Clustering
        if 'clustering' in analysis_types:
            print("7. Text Clustering...")
            if 'tfidf_matrix' in self.results:
                # Use TF-IDF for clustering
                embeddings = self.results['tfidf_matrix'].toarray()
                clusters, kmeans, silhouette = self.clusterer.kmeans_clustering(embeddings)
                self.results['clusters'] = clusters
                self.results['kmeans'] = kmeans
                self.results['silhouette_score'] = silhouette
        
        # Word Cloud
        if 'visualization' in analysis_types:
            print("8. Creating Visualizations...")
            self.analyzer.create_word_cloud(processed_texts)
        
        print("NLP Analysis Completed!")
        return self.results
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        report = {}
        
        if 'statistics' in self.results:
            report['Statistics'] = self.results['statistics']
        
        if 'word_frequency' in self.results:
            report['Top Words'] = self.results['word_frequency'][:10]
        
        if 'sentiment' in self.results:
            sentiment_summary = self.results['sentiment']['sentiment_label'].value_counts()
            report['Sentiment Distribution'] = sentiment_summary.to_dict()
        
        if 'topic_words' in self.results:
            report['Topics'] = {}
            for topic_idx, words in self.results['topic_words']:
                report['Topics'][f'Topic {topic_idx + 1}'] = words
        
        if 'silhouette_score' in self.results:
            report['Clustering Quality'] = {
                'Silhouette Score': self.results['silhouette_score']
            }
        
        return report

# Example usage
if __name__ == "__main__":
    # Sample text data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This is a wonderful day!",
        "Machine learning and artificial intelligence are transforming the world.",
        "I love reading books about science and technology. They are fascinating.",
        "The weather today is terrible. It's raining cats and dogs.",
        "Python programming language is excellent for data science and machine learning.",
        "Natural language processing helps computers understand human language.",
        "Deep learning models achieve state-of-the-art results in many tasks.",
        "The food at that restaurant was absolutely delicious and amazing.",
        "I'm feeling very happy and excited about the upcoming project.",
        "The movie was boring and disappointing. I wouldn't recommend it."
    ]
    
    # Initialize pipeline
    nlp_pipeline = SimpleNLPipeline()
    
    # Run complete analysis
    results = nlp_pipeline.run_complete_analysis(
        sample_texts,
        analysis_types=['preprocessing', 'statistics', 'sentiment', 'topics', 'clustering']
    )
    
    # Generate report
    report = nlp_pipeline.generate_report()
    
    print("\n" + "="*50)
    print("NLP ANALYSIS REPORT")
    print("="*50)
    
    for section, content in report.items():
        print(f"\n{section}:")
        if isinstance(content, dict):
            for key, value in content.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {content}")
    
    # Show some visualizations
    if 'topic_words' in results:
        nlp_pipeline.topic_modeler.visualize_topics(
            results['lda_model'], 
            results['tfidf_vectorizer']
        )
    
    if 'clusters' in results:
        # For visualization, we need 2D embeddings
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(results['tfidf_matrix'].toarray())
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=results['clusters'], cmap='viridis')
        plt.colorbar(scatter)
        plt.title('Text Clusters (PCA)')
        plt.show()

    print("\nSimple NLP pipeline completed successfully!")