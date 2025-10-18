from transformers import pipeline
import os
import re
from dotenv import load_dotenv

load_dotenv()

class AIAnalyzer:
    def __init__(self):
        """Initialize Hugging Face models"""
        print("Loading AI models... This may take a minute on first run...")
        
        try:
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=-1
            )
            
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=-1
            )
            
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
            
            print("✅ AI models loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            raise
    
    def _truncate_text(self, text, max_words):
        """Safely truncate text to max words"""
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return text
    
    def _split_into_chunks(self, text, chunk_size=400):
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        return chunks
    
    def _find_relevant_context(self, question, text, context_size=350):
        """Find the most relevant section of text for the question"""
        # Extract keywords from question
        question_lower = question.lower()
        keywords = [w for w in question_lower.split() if len(w) > 3]
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Score each sentence by keyword matches
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = sum(1 for keyword in keywords if keyword in sentence_lower)
            if score > 0:
                scored_sentences.append((score, i, sentence))
        
        if not scored_sentences:
            # No keyword matches, use first part
            return ' '.join(sentences[:10])
        
        # Sort by score and get top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        # Get the best sentence and surrounding context
        best_idx = scored_sentences[0][1]
        
        # Include sentences before and after for context
        start_idx = max(0, best_idx - 2)
        end_idx = min(len(sentences), best_idx + 3)
        
        context = ' '.join(sentences[start_idx:end_idx])
        
        # Ensure context isn't too long
        words = context.split()
        if len(words) > context_size:
            context = ' '.join(words[:context_size])
        
        return context
    
    def _enhance_answer(self, raw_answer, context, confidence):
        """Enhance the raw answer with surrounding context"""
        # Find the answer in context
        answer_pos = context.lower().find(raw_answer.lower())
        
        if answer_pos == -1:
            return raw_answer
        
        # Get surrounding sentences
        sentences = re.split(r'[.!?]+', context)
        
        # Find which sentence contains the answer
        answer_sentence = None
        for sentence in sentences:
            if raw_answer.lower() in sentence.lower():
                answer_sentence = sentence.strip()
                break
        
        if answer_sentence:
            # Return the full sentence instead of just the fragment
            return answer_sentence + '.'
        
        return raw_answer
    
    def answer_question(self, question, context):
        """Answer questions based on document context - ENHANCED VERSION"""
        try:
            # Find most relevant section
            relevant_context = self._find_relevant_context(question, context)
            
            if len(relevant_context.strip()) < 50:
                return {
                    'answer': "I couldn't find relevant information to answer this question in the document.",
                    'confidence': 0,
                    'context_used': '',
                    'explanation': "The document doesn't seem to contain information related to your question."
                }
            
            # Get answer from model
            result = self.qa_pipeline(
                question=question,
                context=relevant_context
            )
            
            raw_answer = result['answer']
            confidence = result['score']
            
            # Enhance the answer with context
            enhanced_answer = self._enhance_answer(raw_answer, relevant_context, confidence)
            
            # Generate explanation based on confidence
            if confidence > 0.7:
                explanation = "High confidence - The answer was found clearly in the document."
            elif confidence > 0.3:
                explanation = "Medium confidence - The answer is likely correct but may be partially extracted."
            elif confidence > 0.1:
                explanation = "Low confidence - The answer was found but the question wording doesn't match the document exactly. Try rephrasing."
            else:
                explanation = "Very low confidence - The model found a possible answer but isn't certain. The information might not be explicitly stated in the document."
            
            return {
                'answer': enhanced_answer,
                'confidence': confidence,
                'context_used': relevant_context[:200] + '...',
                'explanation': explanation
            }
            
        except Exception as e:
            return {
                'answer': f"Error processing question: {str(e)}",
                'confidence': 0,
                'context_used': '',
                'explanation': "An error occurred while processing your question."
            }
    
    def summarize_text(self, text, max_length=150, min_length=40):
        """Generate summary of text - handles large documents"""
        try:
            words = text.split()
            total_words = len(words)
            
            if total_words < 30:
                return "Text too short to summarize effectively."
            
            if total_words > 500:
                beginning = ' '.join(words[:200])
                middle_start = total_words // 2 - 100
                middle = ' '.join(words[middle_start:middle_start + 200])
                end = ' '.join(words[-100:])
                text_to_summarize = f"{beginning} {middle} {end}"
            else:
                text_to_summarize = text
            
            text_to_summarize = self._truncate_text(text_to_summarize, 450)
            
            summary = self.summarizer(
                text_to_summarize,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            try:
                sentences = text.split('.')[:5]
                return '. '.join(s.strip() for s in sentences if len(s.strip()) > 10) + '.'
            except:
                return f"Document contains {len(text.split())} words. Unable to generate AI summary for very large documents."
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text - handles large documents"""
        try:
            chunks = self._split_into_chunks(text, chunk_size=300)
            
            if not chunks:
                return {
                    'label': 'NEUTRAL',
                    'score': 0.5,
                    'description': 'Document too short to analyze'
                }
            
            sentiments = []
            for chunk in chunks[:3]:
                try:
                    result = self.sentiment_analyzer(chunk)[0]
                    sentiments.append(result)
                except:
                    continue
            
            if not sentiments:
                return {
                    'label': 'ERROR',
                    'score': 0,
                    'description': 'Could not analyze sentiment'
                }
            
            avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
            most_common_label = max(set(s['label'] for s in sentiments), 
                                   key=lambda x: sum(1 for s in sentiments if s['label'] == x))
            
            return {
                'label': most_common_label,
                'score': avg_score,
                'description': self._get_sentiment_description(most_common_label, avg_score)
            }
            
        except Exception as e:
            return {
                'label': 'ERROR',
                'score': 0,
                'description': f"Could not analyze sentiment: {str(e)}"
            }
    
    def extract_key_points(self, text):
        """Extract key sentences from text"""
        try:
            sentences = []
            for line in text.split('\n'):
                for sent in line.split('.'):
                    sent = sent.strip()
                    if 20 < len(sent) < 200:
                        sentences.append(sent)
            
            if not sentences:
                return ["No key points could be extracted from this document."]
            
            total = len(sentences)
            key_indices = [
                0,
                total // 4,
                total // 2,
                3 * total // 4,
                total - 1
            ]
            
            key_points = []
            for idx in key_indices:
                if 0 <= idx < len(sentences):
                    point = sentences[idx].strip()
                    if point and point not in key_points:
                        key_points.append(point + '.')
            
            return key_points[:5] if key_points else ["Document content extracted but key points could not be determined."]
            
        except Exception as e:
            return [f"Error extracting key points: {str(e)}"]
    
    def get_suggested_questions(self, text):
        """Generate suggested questions based on document content"""
        suggestions = []
        first_part = ' '.join(text.split()[:500]).lower()
        
        # Look for specific patterns
        if any(word in first_part for word in ["benefit", "advantage", "positive"]):
            suggestions.append("What are the benefits mentioned?")
        
        if any(word in first_part for word in ["challenge", "problem", "issue", "difficult"]):
            suggestions.append("What challenges are discussed?")
        
        if any(word in first_part for word in ["use", "application", "example"]):
            suggestions.append("What are the practical applications?")
        
        if "career" in first_part or "job" in first_part or "salary" in first_part:
            suggestions.append("What career opportunities are mentioned?")
        
        # Always include generic questions
        if not suggestions:
            suggestions = [
                "What is this document about?",
                "What is the main topic?",
                "What are the key points?"
            ]
        
        return suggestions[:3]
    
    def _get_sentiment_description(self, label, score):
        """Get human-readable sentiment description"""
        if label == 'POSITIVE':
            if score > 0.9:
                return "Very positive/optimistic tone"
            elif score > 0.7:
                return "Positive/informative tone"
            else:
                return "Somewhat positive tone"
        else:
            if score > 0.9:
                return "Very negative/critical tone"
            elif score > 0.7:
                return "Negative/cautionary tone"
            else:
                return "Neutral/formal tone"