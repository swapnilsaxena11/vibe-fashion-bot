from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.0,
                    anthropic_api_key="<your_api_key>")


class UserPreference(BaseModel):
    category: List[str] = []
    sleeve_length: List[str] = []
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    fabric: List[str] = []
    fit: List[str] = []
    size: List[str] = []
    color: List[str] = []
    occasion: List[str] = []
    neckline: List[str] = []
    length: List[str] = []
    pant_type: List[str] = []


llm_with_structure = llm.with_structured_output(UserPreference)


class CustomState(MessagesState):
    question_count: int = 0
    user_preferences: dict = {}


class RAGProductSearch:
    def __init__(self, csv_path: str):
        """Initialize RAG search with product CSV and embedding model."""
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.product_embeddings = None
        self._prepare_embeddings()

    def _prepare_embeddings(self):
        """Create embeddings for all products based on your CSV structure."""
        product_texts = []

        for _, row in self.df.iterrows():
            # Create comprehensive product description for embedding
            text_parts = []

            # Product name and category
            if pd.notna(row['name']):
                text_parts.append(f"Product: {row['name']}")
            if pd.notna(row['category']):
                text_parts.append(f"Category: {row['category']}")

            # Fit and fabric (key style attributes)
            if pd.notna(row['fit']):
                text_parts.append(f"Fit: {row['fit']}")
            if pd.notna(row['fabric']):
                text_parts.append(f"Fabric: {row['fabric']}")

            # Style details
            if pd.notna(row['sleeve_length']):
                text_parts.append(f"Sleeves: {row['sleeve_length']}")
            if pd.notna(row['color_or_print']):
                text_parts.append(f"Color or Print: {row['color_or_print']}")
            if pd.notna(row['occasion']):
                text_parts.append(f"Occasion: {row['occasion']}")

            # Dress-specific attributes
            if pd.notna(row['neckline']):
                text_parts.append(f"Neckline: {row['neckline']}")
            if pd.notna(row['length']):
                text_parts.append(f"Length: {row['length']}")

            # Available sizes
            if pd.notna(row['available_sizes']):
                text_parts.append(f"Available sizes: {row['available_sizes']}")

            product_text = " ".join(text_parts)
            product_texts.append(product_text)

        # Create embeddings
        self.product_embeddings = self.model.encode(product_texts)
        print(f"Created embeddings for {len(product_texts)} products from your fashion catalog")

    def _create_search_query(self, preferences: dict) -> str:
        """Convert user preferences to search query."""
        query_parts = []

        for key, value in preferences.items():
            if value:
                if isinstance(value, list):
                    query_parts.extend([f"{key}: {v}" for v in value])
                else:
                    query_parts.append(f"{key}: {value}")

        return " ".join(query_parts)

    def _apply_filters(self, preferences: dict) -> pd.DataFrame:
        """Apply hard filters based on preferences using your CSV structure."""
        filtered_df = self.df.copy()

        # Price filtering
        if preferences.get('price_min'):
            filtered_df = filtered_df[filtered_df['price'] >= preferences['price_min']]
        if preferences.get('price_max'):
            filtered_df = filtered_df[filtered_df['price'] <= preferences['price_max']]

        # Category filtering (exact and partial matches)
        if preferences.get('category'):
            category_patterns = [cat.lower() for cat in preferences['category']]
            category_mask = filtered_df['category'].str.lower().str.contains(
                '|'.join(category_patterns), na=False, regex=True
            )
            if category_mask.any():
                filtered_df = filtered_df[category_mask]

        # Fabric filtering
        if preferences.get('fabric'):
            fabric_patterns = [fab.lower() for fab in preferences['fabric']]
            fabric_mask = filtered_df['fabric'].str.lower().str.contains(
                '|'.join(fabric_patterns), na=False, regex=True
            )
            if fabric_mask.any():
                filtered_df = filtered_df[fabric_mask]

        # Color filtering
        if preferences.get('color'):
            color_patterns = [col.lower() for col in preferences['color']]
            color_mask = filtered_df['color_or_print'].str.lower().str.contains(
                '|'.join(color_patterns), na=False, regex=True
            )
            if color_mask.any():
                filtered_df = filtered_df[color_mask]

        # Fit filtering
        if preferences.get('fit'):
            fit_patterns = [fit.lower() for fit in preferences['fit']]
            fit_mask = filtered_df['fit'].str.lower().str.contains(
                '|'.join(fit_patterns), na=False, regex=True
            )
            if fit_mask.any():
                filtered_df = filtered_df[fit_mask]

        # Occasion filtering
        if preferences.get('occasion'):
            occasion_patterns = [occ.lower() for occ in preferences['occasion']]
            occasion_mask = filtered_df['occasion'].str.lower().str.contains(
                '|'.join(occasion_patterns), na=False, regex=True
            )
            if occasion_mask.any():
                filtered_df = filtered_df[occasion_mask]

        # Size filtering (check if size is available)
        if preferences.get('size'):
            size_patterns = [size.upper() for size in preferences['size']]  # Sizes are typically uppercase
            size_mask = pd.Series([False] * len(filtered_df))

            for idx, available_sizes in enumerate(filtered_df['available_sizes']):
                if pd.notna(available_sizes):
                    available_str = str(available_sizes).upper()
                    if any(size in available_str for size in size_patterns):
                        size_mask.iloc[idx] = True

            if size_mask.any():
                filtered_df = filtered_df[size_mask]

        return filtered_df

    def search_products(self, preferences: dict, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for products using RAG approach."""
        # Apply hard filters first
        filtered_df = self._apply_filters(preferences)

        if filtered_df.empty:
            # Fallback to full dataset if filters are too restrictive
            filtered_df = self.df.copy()

        # Create search query from preferences
        search_query = self._create_search_query(preferences)

        if not search_query.strip():
            # Return random sample if no preferences
            sample_df = filtered_df.sample(min(top_k, len(filtered_df)))
            return sample_df.to_dict('records')

        # Get query embedding
        query_embedding = self.model.encode([search_query])

        # Get embeddings for filtered products
        filtered_indices = filtered_df.index.tolist()
        filtered_embeddings = self.product_embeddings[filtered_indices]

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]

        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return results with similarity scores
        results = []
        for idx in top_indices:
            product_idx = filtered_indices[idx]
            product = filtered_df.iloc[idx].to_dict()
            product['similarity_score'] = float(similarities[idx])
            results.append(product)

        return results


def extract_preferences(state):
    """Extract preferences from all messages so far."""
    messages = state["messages"]
    current_prefs = state.get("user_preferences", {})

    # Get all human messages (user responses)
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]

    if user_messages:
        try:
            extract_prompt = f"""
            Based on these user messages about fashion preferences, extract what they mentioned.Your are free to infer preferences from the user messages.
            Current stored preferences: {current_prefs}

            Extract only explicit preferences:
            - Categories (dress, top, jeans, etc.)
            - Sizes mentioned
            - Price ranges mentioned  
            - Specific features (sleeveless, colors, etc.)
            - Occasions mentioned
            - Fabric types mentioned
            - Fit preferences (loose, fitted, etc.)
            - Neckline types (if applicable)
            - Length preferences (if applicable)
            - Pant types (if applicable)
            - Any other relevant fashion preferences
            

            Return updated preferences, merging with existing ones.
            """

            updated_preference = llm_with_structure.invoke([
                                                               SystemMessage(content=extract_prompt)
                                                           ] + messages)

            # Merge preferences
            merged_prefs = current_prefs.copy()
            for key, value in updated_preference.model_dump().items():
                if value is not None:
                    if isinstance(value, list) and value:
                        if key in merged_prefs and isinstance(merged_prefs[key], list):
                            combined = merged_prefs[key] + value
                            merged_prefs[key] = list(set(combined))
                        else:
                            merged_prefs[key] = value
                    elif not isinstance(value, list):
                        merged_prefs[key] = value

            return {
                "messages": messages,
                "question_count": state.get("question_count", 0),
                "user_preferences": merged_prefs
            }

        except Exception as e:
            print(f"Debug - Error extracting preferences: {e}")

    return state


def ask_question(state):
    """Ask the next question based on current state."""
    messages = state["messages"]
    question_count = state.get("question_count", 0)

    if question_count < 3:
        system_prompt = f"""You are fashVibey, a friendly AI fashion consultant.
        question_count: {question_count}
        preferences: {state.get("user_preferences", {})}

        -Ask questions to understand the user's fashion preferences better. 
        -Avoid asking about preferences which you've already inferred or which have been provided by the user.
        -If on the 3rd question, ask for any missing details, preferences which you've not mapped yet.
        -The questions should not sound like an intense questionaire, but rather a friendly conversation.

        Make it feel personal and build on what they've already shared. Keep it warm and friendly."""

        response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    else:
        response = AIMessage(content="Let me help you find the perfect items!")

    return {
        "messages": messages + [response],
        "question_count": question_count + 1,
        "user_preferences": state.get("user_preferences", {})
    }


def create_final_mapping_and_search(state):
    """Create comprehensive final mapping and search for products."""
    messages = state["messages"]
    user_prefs = state.get("user_preferences", {})

    try:
        # Get all user messages for comprehensive analysis
        user_messages = [msg.content for msg in messages if isinstance(msg, HumanMessage)]

        # Create comprehensive mapping with inferences
        mapping_prompt = f"""
        Based on this fashion consultation, create a comprehensive preference mapping.

        Extracted preferences: {user_prefs}

        Rules:
        1. Include explicit preferences mentioned by the user
        2. Intelligently infer related preferences:
           - "summer" → breathable fabrics like linen, cotton
           - "casual" → relaxed fit
           - "beach" → flowy, light colors, breathable fabrics
           - "brunch" → smart-casual, comfortable but put-together
        3. Use fashion expertise to complete the picture
        4. Only include meaningful, non-empty values

        Create a complete, realistic preference mapping.
        """

        final_mapping = llm_with_structure.invoke([SystemMessage(content=mapping_prompt)] + messages)

        # Clean up the mapping
        clean_mapping = {}
        for key, value in final_mapping.model_dump().items():
            if value is not None:
                if isinstance(value, list) and value:
                    clean_mapping[key] = value
                elif not isinstance(value, list) and value:
                    clean_mapping[key] = value

        # Search for products using RAG
        try:
            # Uncomment the next line when you have your CSV ready
            search_results = rag_search.search_products(clean_mapping, top_k=10)

            # For demonstration with your CSV structure, here's what the results would look like:
            search_results = [
                {"id": "T001", "name": "Sun-Dappled Serenity top", "category": "top", "price": 72, "fabric": "Linen",
                 "color": "Pastel yellow", "similarity_score": 0.89},
                {"id": "D002", "name": "Ripple Linen dress", "category": "dress", "price": 95, "fabric": "Linen",
                 "color": "Seafoam green", "similarity_score": 0.85},
                {"id": "T006", "name": "Cloud-Soft Harmony top", "category": "top", "price": 42,
                 "fabric": "Modal jersey", "color": "Pastel pink", "similarity_score": 0.82},
                {"id": "D007", "name": "Breeze Blush dress", "category": "dress", "price": 120, "fabric": "Cotton",
                 "color": "Blush pink", "similarity_score": 0.80},
                {"id": "D015", "name": "Sage Whisper dress", "category": "dress", "price": 98, "fabric": "Linen blend",
                 "color": "Sage green", "similarity_score": 0.78},
            ]

            # Extract product IDs
            product_ids = [product["id"] for product in search_results]

            # Create detailed product recommendations with your CSV format
            product_details = "\n".join([
                f"• {product['name']} (ID: {product['id']}) - ${product['price']}\n"
                f"  {product['category'].title()} | {product['fabric']} | {product['color']}\n"
                f"  Match Score: {product['similarity_score']:.2f}"
                for product in search_results[:5]
            ])

        except Exception as search_error:
            print(f"Search error: {search_error}")
            product_ids = []
            product_details = "Sorry, I couldn't search the product catalog right now."

        # Create justification
        justification_prompt = f"""
        Write a warm, personal justification for these style recommendations.

        User's requests: {user_messages}
        Final recommendations: {clean_mapping}
        Found products: {len(product_ids)} items

        Write a friendly explanation that:
        1. References their specific vibe/request from the first message
        2. Explains key inferences you made and why
        3. Shows how recommendations match their needs
        4. Mentions the products found for them

        Keep it conversational and personal!
        """

        justification = llm.invoke([HumanMessage(content=justification_prompt)])

        # Format final response with product recommendations
        result_text = f"""**Your Fashion Recommendations**

{justification.content}

**Recommended Products:**
{product_details}

**Product IDs for your reference:**
{', '.join(product_ids) if product_ids else 'No products found'}

**Search Criteria Used:**
```json
{json.dumps(clean_mapping, indent=2)}
```
"""

        return {
            "messages": messages + [AIMessage(content=result_text)],
            "question_count": state.get("question_count", 0),
            "user_preferences": clean_mapping,
            "product_ids": product_ids  # Store product IDs in state
        }

    except Exception as e:
        return {
            "messages": messages + [AIMessage(content=f"Sorry, there was an error: {e}")],
            "question_count": state.get("question_count", 0),
            "user_preferences": user_prefs,
            "product_ids": []
        }


def should_continue(state):
    """Route to next step based on question count."""
    question_count = state.get("question_count", 0)
    if question_count < 3:
        return "ask_question"
    else:
        return "create_final_mapping_and_search"


# Build the graph
builder = StateGraph(CustomState)

# Add nodes
builder.add_node("extract_preferences", extract_preferences)
builder.add_node("ask_question", ask_question)
builder.add_node("create_final_mapping_and_search", create_final_mapping_and_search)

# Set up the flow
builder.add_edge(START, "extract_preferences")
builder.add_conditional_edges("extract_preferences", should_continue)
builder.add_edge("ask_question", "extract_preferences")
builder.add_edge("create_final_mapping_and_search", END)

# Compile with interrupt before asking questions
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["ask_question"])

# Config for memory
config = {"configurable": {"thread_id": "fashion_chat"}}


def run_conversation():
    """Run the conversation with proper interrupt handling."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if initialize_rag_search(f"{script_dir}/Apparels_shared.csv"):
        print("Welcome to fashVibey! Tell me about your style vibe and I'll help you find the perfect items.")
        print("I'll ask you a few questions to understand your preferences better.\n")
    else:
        print("Error initializing RAG search. Please check your CSV path and format.")
        return

    # Get initial input
    initial_input = input("Shopper: ").strip()
    if not initial_input:
        print("Please tell me about your style!")
        return

    try:
        # Start with initial message
        initial_state = {"messages": [HumanMessage(content=initial_input)]}

        # Run initial processing
        for event in graph.stream(initial_state, config, stream_mode="values"):
            pass

        # Handle conversation loop with interrupts
        while True:
            snapshot = graph.get_state(config)

            if snapshot.next:
                # Continue to get the question
                for event in graph.stream(None, config, stream_mode="values"):
                    current_state = event

                # Show the question
                if "messages" in current_state:
                    last_message = current_state["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        print(f"Agent: {last_message.content}")

                        # Get user response
                        user_input = input("\nShopper: ").strip()
                        if not user_input:
                            print("Please provide an answer!")
                            continue

                        # Continue with user input
                        user_state = {"messages": [HumanMessage(content=user_input)]}
                        for event in graph.stream(user_state, config, stream_mode="values"):
                            pass
            else:
                # Get final result with product recommendations
                final_snapshot = graph.get_state(config)
                if final_snapshot.values.get("messages"):
                    last_message = final_snapshot.values["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        print(f"Agent: {last_message.content}")

                        # Show product IDs if available
                        if final_snapshot.values.get("product_ids"):
                            print(f"\nProduct IDs found: {final_snapshot.values['product_ids']}")
                break

    except KeyboardInterrupt:
        print("\nConversation ended.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def reset_conversation():
    """Reset conversation with new thread ID."""
    global config
    import uuid
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}


def initialize_rag_search(csv_path: str):
    """Initialize the RAG search with your product CSV."""
    global rag_search
    try:
        rag_search = RAGProductSearch(csv_path)
        print(f"✅ RAG search initialized successfully!")
        print(f"📊 Loaded {len(rag_search.df)} products from your fashion catalog")
        print(f"🏷️  Categories available: {rag_search.df['category'].unique()}")
        return True
    except Exception as e:
        print(f"❌ Error initializing RAG search: {e}")
        return False


if __name__ == "__main__":

    while True:
        print("\nOptions:")
        print("1. Start new conversation")
        print("2. Reset and start fresh")
        print("3. Quit")

        choice = input("Choose (1-3): ").strip()

        if choice == '1':
            run_conversation()
        elif choice == '2':
            reset_conversation()
            print("Starting fresh...")
            run_conversation()
        elif choice == '3':
            break
        else:
            print("Please choose 1, 2 or 3")
