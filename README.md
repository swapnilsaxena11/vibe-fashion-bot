# FashVibey - AI Fashion Consultant with RAG Search 🛍️✨

An intelligent fashion chatbot that understands your style preferences through natural conversation and recommends products from your catalog using advanced RAG (Retrieval-Augmented Generation) search.

## 🌟 Features

- **Conversational AI**: Natural, friendly fashion consultation through 3 contextual questions
- **Smart Preference Extraction**: Automatically extracts and infers fashion preferences from user input
- **RAG-Powered Search**: Semantic search through your product catalog using embeddings
- **Intelligent Filtering**: Combines exact filters with similarity matching for better results
- **Memory & State Management**: Maintains conversation context using LangGraph
- **Product Recommendations**: Returns actual product IDs with detailed justifications

## 🏗️ Architecture

```
User Input → Preference Extraction → Contextual Questions → Final Mapping → RAG Search → Product IDs
     ↓              ↓                      ↓                  ↓            ↓
   LangChain    Pydantic Models      LangGraph State     Embeddings   Product Results
```

## 📋 Prerequisites

- Python 3.8+
- Anthropic API key
- Product catalog in CSV format

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone vibe-sahion-bot
cd vibe-sahion-bot
```

### 2. Install Dependencies
```bash
pip install sentence-transformers scikit-learn pandas numpy langchain-anthropic langgraph pydantic
```

### 3. Set Up API Key
```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```
Or update the API key directly in the code:
```python
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.0,
                    anthropic_api_key="your_api_key_here")
```

## 📊 CSV Data Format

Your product catalog should have these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `id` | Unique product identifier | T001, D002 |
| `name` | Product name | "Sun-Dappled Serenity top" |
| `category` | Product type | top, dress |
| `available` | Available sizes | XS,S,M,L,XL |
| `fit` | Fit style | Relaxed, Body-hugging, Flowy |
| `fabric` | Material | Linen, Silk, Cotton, Modal |
| `sleeve_length` | Sleeve type | Short, Long sleeves, Sleeveless |
| `color` | Color description | Pastel yellow, Deep blue |
| `occasion` | Suitable occasions | Party, Vacation, Everyday |
| `neckline` | Neckline style | V neck, Square neck, Sweetheart |
| `length` | Length description | Short flutter, Long sleeves |
| `pant_type` | For pants/bottoms | (if applicable) |
| `price` | Product price | 72, 95, 120 |

### Sample CSV Structure:
```csv
id,name,category,available,fit,fabric,sleeve_length,color,occasion,neckline,length,pant_type,price
T001,Sun-Dappled Serenity top,top,XS-S-M-L,Relaxed,Linen,Short,Pastel yellow,,,,,72
D002,Ripple Linen dress,dress,XS-S-M-L,Relaxed,Linen,Sleeveless,Seafoam green,Vacation,Square neck,,,95
```

## 🎯 Usage

### 1. Initialize the System
```python
python fash_vibey.py
```


```

### 2. Start a Conversation
```
Choose option 1: "Start new conversation"
```

### 3. Example Conversation Flow
```
Welcome to fashVibey! Tell me about your style vibe...

Shopper: I want something casual and flowy for summer brunch

Agent: That sounds lovely! For your summer brunch vibe, are you thinking more of a dress or separates like a top and bottoms?

Shopper: I'd prefer a dress

Agent: Perfect! What's your usual size, and do you have a preferred price range in mind?

Shopper: Size Medium, and under $120 would be great

Agent: Great! One last question - are there any colors you're particularly drawn to or want to avoid?

Shopper: I love pastels and earth tones

Agent: **Your Fashion Recommendations**

Based on your casual summer brunch vibe, I selected flowy, breathable fabrics in your favorite pastel and earth tone palette that perfectly match your relaxed yet put-together style!

**Recommended Products:**
• Ripple Linen dress (ID: D002) - $95
  Dress | Linen | Seafoam green
  Match Score: 0.85

• Breeze Blush dress (ID: D007) - $120  
  Dress | Cotton | Blush pink
  Match Score: 0.80

**Product IDs for your reference:**
D002, D007, D015, T006, D017
```

## 🔧 Configuration

### Customize Questions
Modify the `ask_question` function to change the conversation flow:
```python
system_prompt = f"""You are fashVibey, a friendly AI fashion consultant.
question_count: {question_count}

-Ask questions to understand the user's fashion preferences better.
-Customize your approach based on user responses
-Keep it warm and friendly."""
```

### Adjust Search Parameters
In `create_final_mapping_and_search`:
```python
search_results = rag_search.search_products(clean_mapping, top_k=10)  # Change top_k for more/fewer results
```

### Model Configuration
Change the embedding model in `RAGProductSearch`:
```python
self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Try 'all-mpnet-base-v2' for better quality
```

## 🧠 How RAG Search Works

### 1. **Embedding Creation**
- Converts each product into rich text: "Product: Sun-Dappled Serenity top Category: top Fit: Relaxed Fabric: Linen..."
- Creates semantic embeddings using SentenceTransformer

### 2. **Smart Filtering**
- **Hard Filters**: Exact matches for price, size availability, specific categories
- **Soft Filters**: Partial matching for colors, fabrics, occasions
- **Fallback Logic**: Broadens search if filters are too restrictive

### 3. **Semantic Matching**
- Converts user preferences to search query
- Calculates cosine similarity between query and product embeddings  
- Returns top-K most similar products with confidence scores

### 4. **Intelligent Inference**
- "Summer" → breathable fabrics (linen, cotton)
- "Casual" → relaxed fits
- "Brunch" → smart-casual, comfortable styles
- "Beach" → flowy, light colors
## 🔍 Troubleshooting

### Common Issues

**1. Import Error for sentence-transformers:**
```bash
pip install --upgrade sentence-transformers torch transformers
```

**2. API Key Issues:**
```bash
export ANTHROPIC_API_KEY="your_key_here"
# Or set it directly in the code
```

**3. CSV Loading Problems:**
- Check column names match exactly
- Ensure CSV is UTF-8 encoded
- Verify no missing required columns

**4. No Products Found:**
- Check if CSV path is correct
- Verify price ranges aren't too restrictive
- Try broader search terms

### Debug Mode
Add debug prints in `RAGProductSearch.search_products()`:
```python
print(f"Original products: {len(self.df)}")
print(f"After filtering: {len(filtered_df)}")
print(f"Search query: {search_query}")
```

## 🚀 Advanced Features

### Custom Preference Models
Extend the `UserPreference` class:
```python
class UserPreference(BaseModel):
    category: List[str] = []
    style_vibe: List[str] = []  # Add custom fields
    sustainability: Optional[bool] = None
    # ... existing fields
```
```

### Custom Scoring
Implement custom similarity scoring in `RAGProductSearch`:
```python
def custom_score(self, product, preferences):
    # Implement your custom scoring logic
    base_score = cosine_similarity_score
    price_boost = 1.0 if product['price'] < preferences.get('price_max', 1000) else 0.8
    return base_score * price_boost
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Anthropic** for Claude AI capabilities
- **LangChain** for conversation management
- **Sentence-Transformers** for semantic search
- **Hugging Face** for transformer models

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Check the troubleshooting section
- Review example conversations in `/examples`

---

**Made with ❤️ for fashion enthusiasts and AI developers**

*Transform your product catalog into an intelligent shopping experience!*
