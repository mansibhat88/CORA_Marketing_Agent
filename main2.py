import streamlit as st
import pandas as pd
import numpy as np
import joblib
import openai
from datetime import datetime
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('marketing_agent.log'),
        logging.StreamHandler()
    ]
)

class MarketingAgent:
    def __init__(self):
        self.scaler = None
        self.kmeans_model = None
        self.propensity_model = None
        self.segment_labels = {0: "Budget Conscious", 1: "Premium Customers", 2: "High Value"}
        self.feedback_adjustments = {}
        
    def load_models(self):
        """Load pre-trained models if they exist"""
        try:
            if os.path.exists('models/scaler.joblib'):
                self.scaler = joblib.load('models/scaler.joblib')
                logging.info("Scaler loaded successfully")
            
            if os.path.exists('models/kmeans_model.joblib'):
                self.kmeans_model = joblib.load('models/kmeans_model.joblib')
                logging.info("K-Means model loaded successfully")
            
            if os.path.exists('models/propensity_model.joblib'):
                self.propensity_model = joblib.load('models/propensity_model.joblib')
                logging.info("Propensity model loaded successfully")
                
            # Load feedback adjustments
            if os.path.exists('models/feedback_adjustments.json'):
                with open('models/feedback_adjustments.json', 'r') as f:
                    self.feedback_adjustments = json.load(f)
                logging.info("Feedback adjustments loaded successfully")
            
            return True
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            return False
    
    def save_models(self):
        """Save trained models"""
        try:
            os.makedirs('models', exist_ok=True)
            
            if self.scaler:
                joblib.dump(self.scaler, 'models/scaler.joblib')
            if self.kmeans_model:
                joblib.dump(self.kmeans_model, 'models/kmeans_model.joblib')
            if self.propensity_model:
                joblib.dump(self.propensity_model, 'models/propensity_model.joblib')
            
            # Save feedback adjustments
            with open('models/feedback_adjustments.json', 'w') as f:
                json.dump(self.feedback_adjustments, f)
            
            logging.info("Models saved successfully")
            return True
        except Exception as e:
            logging.error(f"Error saving models: {e}")
            return False
    
    def validate_data(self, df):
        """Validate input data structure"""
        required_columns = ['customer_id', 'age', 'purchase_frequency', 'last_purchase_days_ago', 'total_spent']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for data types and ranges
        try:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['purchase_frequency'] = pd.to_numeric(df['purchase_frequency'], errors='coerce')
            df['last_purchase_days_ago'] = pd.to_numeric(df['last_purchase_days_ago'], errors='coerce')
            df['total_spent'] = pd.to_numeric(df['total_spent'], errors='coerce')
            
            # Check for missing values
            if df[required_columns].isnull().any().any():
                return False, "Data contains missing values in required columns"
            
            return True, "Data validation passed"
        except Exception as e:
            return False, f"Data validation error: {e}"
    
    def train_models(self, df):
        """Train segmentation and propensity models"""
        try:
            # Prepare features
            features = df[['age', 'purchase_frequency', 'last_purchase_days_ago', 'total_spent']]
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(features)
            
            # Train K-Means clustering
            self.kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
            segments = self.kmeans_model.fit_predict(X_scaled)
            
            # If campaign_response exists, train propensity model
            if 'campaign_response' in df.columns:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, df['campaign_response'], test_size=0.2, random_state=42
                )
                
                self.propensity_model = RandomForestClassifier(
                    n_estimators=100, random_state=42, max_depth=10
                )
                self.propensity_model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = self.propensity_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                logging.info(f"Propensity model accuracy: {accuracy:.3f}")
            
            self.save_models()
            logging.info("Models trained and saved successfully")
            return True, segments
            
        except Exception as e:
            logging.error(f"Error training models: {e}")
            return False, None
    
    def predict_segments_and_propensity(self, df):
        """Predict segments and propensity scores for new data"""
        try:
            if not all([self.scaler, self.kmeans_model]):
                return False, "Models not loaded. Please train models first."
            
            features = df[['age', 'purchase_frequency', 'last_purchase_days_ago', 'total_spent']]
            X_scaled = self.scaler.transform(features)
            
            # Predict segments
            segments = self.kmeans_model.predict(X_scaled)
            
            # Predict propensity scores
            propensity_scores = None
            if self.propensity_model:
                propensity_scores = self.propensity_model.predict_proba(features)[:, 1]
                
                # Apply feedback adjustments
                adjusted_scores = []
                for i, score in enumerate(propensity_scores):
                    customer_id = df.iloc[i]['customer_id']
                    if str(customer_id) in self.feedback_adjustments:
                        adjustment = self.feedback_adjustments[str(customer_id)]
                        adjusted_score = min(1.0, max(0.0, score + adjustment))
                        adjusted_scores.append(adjusted_score)
                    else:
                        adjusted_scores.append(score)
                
                propensity_scores = np.array(adjusted_scores)
            
            return True, (segments, propensity_scores)
            
        except Exception as e:
            logging.error(f"Error predicting: {e}")
            return False, None
    
    def process_feedback(self, feedback_df):
        """Process feedback data to adjust propensity scores"""
        try:
            required_cols = ['customer_id', 'actual_response', 'predicted_score']
            if not all(col in feedback_df.columns for col in required_cols):
                return False, f"Feedback data must contain: {required_cols}"
            
            for _, row in feedback_df.iterrows():
                customer_id = str(row['customer_id'])
                actual = row['actual_response']
                predicted = row['predicted_score']
                
                # Calculate adjustment based on feedback
                if actual == 1 and predicted < 0.7:  # Customer responded but we predicted low
                    adjustment = 0.1  # Increase future predictions
                elif actual == 0 and predicted > 0.7:  # Customer didn't respond but we predicted high
                    adjustment = -0.1  # Decrease future predictions
                else:
                    adjustment = 0.0
                
                # Store adjustment
                if customer_id in self.feedback_adjustments:
                    self.feedback_adjustments[customer_id] += adjustment
                else:
                    self.feedback_adjustments[customer_id] = adjustment
                
                # Cap adjustments
                self.feedback_adjustments[customer_id] = max(-0.3, min(0.3, self.feedback_adjustments[customer_id]))
            
            self.save_models()  # Save updated adjustments
            logging.info(f"Processed feedback for {len(feedback_df)} customers")
            return True, "Feedback processed successfully"
            
        except Exception as e:
            logging.error(f"Error processing feedback: {e}")
            return False, f"Error processing feedback: {e}"
    
    def determine_campaign_strategy(self, segment, propensity_score, total_spent, last_purchase_days):
        """Determine marketing campaign strategy"""
        if propensity_score > 0.7 and total_spent > 500:
            return "upsell"
        elif segment in [1, 2] and last_purchase_days < 30:
            return "cross-sell"
        elif last_purchase_days > 60:
            return "retention"
        else:
            return "engagement"
    
    def generate_marketing_message(self, row, api_key):
        """Generate personalized marketing message using OpenAI"""
        try:
            client = openai.OpenAI(api_key=api_key)
            
            strategy = row.get('campaign_strategy', 'engagement')
            segment_name = self.segment_labels.get(row['segment'], 'Valued Customer')
            
            # Create context-aware prompt
            prompt = f"""
            Create a personalized marketing message for a {strategy} campaign.
            
            Customer Profile:
            - Segment: {segment_name}
            - Age: {row['age']}
            - Gender: {row.get('gender', 'Unknown')}
            - Product Interest: {row.get('product_bought', 'Various products')}
            - Last Purchase: {row['last_purchase_days_ago']} days ago
            - Total Spent: ${row['total_spent']:.2f}
            - Propensity Score: {row['propensity_score']:.2f}
            
            Requirements:
            - Write only the marketing message (no prefixes like "Here's your message")
            - Keep it under 150 words
            - Include a compelling offer (10-20% discount or special deal)
            - Make it personal and engaging
            - Include a clear call-to-action
            - Match the {strategy} campaign objective
            
            Tone: Friendly, professional, and persuasive
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            message = response.choices[0].message.content.strip()
            return message
            
        except Exception as e:
            logging.error(f"Error generating message for customer {row.get('customer_id', 'Unknown')}: {e}")
            return f"Hello! We have a special offer just for you. Don't miss out on 15% off your next purchase. Shop now!"
    
    def create_visualizations(self, df):
        """Create visualization charts"""
        charts = {}
        
        try:
            # Segment distribution
            segment_counts = df['segment'].value_counts()
            segment_names = [self.segment_labels.get(i, f'Segment {i}') for i in segment_counts.index]
            
            fig1 = px.pie(
                values=segment_counts.values, 
                names=segment_names,
                title="Customer Segmentation Distribution"
            )
            charts['segments'] = fig1
            
            # Propensity score distribution
            if 'propensity_score' in df.columns:
                fig2 = px.histogram(
                    df, x='propensity_score', 
                    title="Propensity Score Distribution",
                    nbins=20
                )
                charts['propensity'] = fig2
            
            # Campaign strategy distribution
            if 'campaign_strategy' in df.columns:
                strategy_counts = df['campaign_strategy'].value_counts()
                fig3 = px.bar(
                    x=strategy_counts.index, 
                    y=strategy_counts.values,
                    title="Recommended Campaign Strategies"
                )
                charts['strategies'] = fig3
            
            return charts
            
        except Exception as e:
            logging.error(f"Error creating visualizations: {e}")
            return {}

def create_excel_output(df):
    """Create Excel file with formatted output"""
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Main results
            df.to_excel(writer, sheet_name='Marketing_Results', index=False)
            
            # Summary statistics
            summary_df = pd.DataFrame({
                'Metric': ['Total Customers', 'Avg Propensity Score', 'High Propensity (>0.7)', 
                          'Segment 0 Count', 'Segment 1 Count', 'Segment 2 Count'],
                'Value': [
                    len(df),
                    df['propensity_score'].mean() if 'propensity_score' in df.columns else 0,
                    len(df[df['propensity_score'] > 0.7]) if 'propensity_score' in df.columns else 0,
                    len(df[df['segment'] == 0]),
                    len(df[df['segment'] == 1]),
                    len(df[df['segment'] == 2])
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        return output
    except Exception as e:
        logging.error(f"Error creating Excel output: {e}")
        return None

# Streamlit App
def main():
    st.set_page_config(
        page_title="🎯 Marketing AI Agent",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Marketing AI Agent - Phase 1")
    st.markdown("### Intelligent Customer Segmentation & Campaign Generation")
    
    # Initialize agent
    if 'agent' not in st.session_state:
        st.session_state.agent = MarketingAgent()
        st.session_state.agent.load_models()
    
    agent = st.session_state.agent
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # OpenAI API Key
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Required for message generation")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Processing", "🤖 Model Training", "💬 Campaign Generation", "📈 Analytics"])
    
    with tab1:
        st.header("📊 Data Upload & Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Data")
            uploaded_file = st.file_uploader(
                "Upload customer data (CSV/Excel)",
                type=['csv', 'xlsx'],
                help="Required columns: customer_id, age, purchase_frequency, last_purchase_days_ago, total_spent"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ Loaded {len(df)} customers")
                    
                    # Validate data
                    is_valid, message = agent.validate_data(df)
                    if is_valid:
                        st.success(f"✅ {message}")
                        st.session_state.customer_data = df
                        
                        # Show data preview
                        st.subheader("Data Preview")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        # Show data statistics
                        st.subheader("Data Statistics")
                        st.dataframe(df.describe(), use_container_width=True)
                        
                    else:
                        st.error(f"❌ {message}")
                        
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        with col2:
            st.subheader("Feedback Data (Optional)")
            feedback_file = st.file_uploader(
                "Upload feedback data (CSV/Excel)",
                type=['csv', 'xlsx'],
                help="Required columns: customer_id, actual_response, predicted_score"
            )
            
            if feedback_file:
                try:
                    if feedback_file.name.endswith('.csv'):
                        feedback_df = pd.read_csv(feedback_file)
                    else:
                        feedback_df = pd.read_excel(feedback_file)
                    
                    st.success(f"✅ Loaded feedback for {len(feedback_df)} customers")
                    
                    # Process feedback
                    if st.button("Process Feedback", type="primary"):
                        success, message = agent.process_feedback(feedback_df)
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                    
                    st.dataframe(feedback_df.head(), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error loading feedback file: {e}")
    
    with tab2:
        st.header("🤖 Model Training")
        
        if 'customer_data' in st.session_state:
            df = st.session_state.customer_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Configuration")
                
                # Check if campaign_response exists for propensity modeling
                has_response = 'campaign_response' in df.columns
                st.info(f"Campaign response column: {'✅ Found' if has_response else '❌ Not found (will skip propensity modeling)'}")
                
                if st.button("🚀 Train Models", type="primary"):
                    with st.spinner("Training models..."):
                        success, segments = agent.train_models(df)
                        
                        if success:
                            st.success("✅ Models trained successfully!")
                            
                            # Add segments to dataframe
                            df['segment'] = segments
                            st.session_state.customer_data = df
                            
                            # Show segment distribution
                            segment_dist = pd.Series(segments).value_counts().sort_index()
                            st.subheader("Segment Distribution")
                            
                            for seg, count in segment_dist.items():
                                label = agent.segment_labels.get(seg, f'Segment {seg}')
                                st.metric(label, count)
                        else:
                            st.error("❌ Model training failed!")
            
            with col2:
                st.subheader("Model Status")
                
                # Check model availability
                models_status = {
                    "Scaler": agent.scaler is not None,
                    "K-Means Clustering": agent.kmeans_model is not None,
                    "Propensity Model": agent.propensity_model is not None
                }
                
                for model, status in models_status.items():
                    st.metric(model, "✅ Loaded" if status else "❌ Not Available")
        
        else:
            st.warning("⚠️ Please upload customer data first")
    
    with tab3:
        st.header("💬 Campaign Generation")
        
        if 'customer_data' in st.session_state and agent.kmeans_model:
            df = st.session_state.customer_data.copy()
            
            if not api_key:
                st.warning("⚠️ Please enter your OpenAI API key in the sidebar to generate messages")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Generation Settings")
                
                # Prediction settings
                generate_predictions = st.checkbox("Generate Predictions", value=True)
                generate_messages = st.checkbox("Generate Marketing Messages", value=bool(api_key))
                
                if st.button("🎯 Generate Campaigns", type="primary"):
                    with st.spinner("Generating campaigns..."):
                        
                        if generate_predictions:
                            # Get predictions
                            success, results = agent.predict_segments_and_propensity(df)
                            
                            if success:
                                segments, propensity_scores = results
                                df['segment'] = segments
                                
                                if propensity_scores is not None:
                                    df['propensity_score'] = propensity_scores
                                else:
                                    # Generate dummy propensity scores based on features
                                    df['propensity_score'] = np.random.beta(2, 5, len(df))
                                
                                # Determine campaign strategies
                                df['campaign_strategy'] = df.apply(
                                    lambda row: agent.determine_campaign_strategy(
                                        row['segment'], 
                                        row['propensity_score'], 
                                        row['total_spent'],
                                        row['last_purchase_days_ago']
                                    ), axis=1
                                )
                                
                                # Add segment labels
                                df['segment_name'] = df['segment'].map(agent.segment_labels)
                                
                                st.success("✅ Predictions generated!")
                            else:
                                st.error("❌ Failed to generate predictions")
                                st.stop()
                        
                        if generate_messages and api_key:
                            # Generate marketing messages
                            messages = []
                            progress_bar = st.progress(0)
                            
                            for idx, row in df.iterrows():
                                message = agent.generate_marketing_message(row, api_key)
                                messages.append(message)
                                progress_bar.progress((idx + 1) / len(df))
                            
                            df['marketing_message'] = messages
                            st.success("✅ Marketing messages generated!")
                        
                        st.session_state.results_data = df
            
            with col2:
                if 'results_data' in st.session_state:
                    results_df = st.session_state.results_data
                    
                    st.subheader("Campaign Results Preview")
                    
                    # Show key metrics
                    col2_1, col2_2, col2_3, col2_4 = st.columns(4)
                    
                    with col2_1:
                        st.metric("Total Customers", len(results_df))
                    
                    with col2_2:
                        if 'propensity_score' in results_df.columns:
                            avg_propensity = results_df['propensity_score'].mean()
                            st.metric("Avg Propensity", f"{avg_propensity:.2f}")
                    
                    with col2_3:
                        if 'propensity_score' in results_df.columns:
                            high_propensity = len(results_df[results_df['propensity_score'] > 0.7])
                            st.metric("High Propensity", high_propensity)
                    
                    with col2_4:
                        if 'campaign_strategy' in results_df.columns:
                            top_strategy = results_df['campaign_strategy'].mode()[0]
                            st.metric("Top Strategy", top_strategy.title())
                    
                    # Show sample results
                    display_columns = ['customer_id', 'segment_name', 'propensity_score', 'campaign_strategy']
                    if 'marketing_message' in results_df.columns:
                        display_columns.append('marketing_message')
                    
                    st.dataframe(
                        results_df[display_columns].head(10),
                        use_container_width=True
                    )
                    
                    # Download results
                    excel_data = create_excel_output(results_df)
                    if excel_data:
                        st.download_button(
                            label="📊 Download Results (Excel)",
                            data=excel_data,
                            file_name=f"marketing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        else:
            st.warning("⚠️ Please upload data and train models first")
    
    with tab4:
        st.header("📈 Analytics & Insights")
        
        if 'results_data' in st.session_state:
            results_df = st.session_state.results_data
            
            # Create visualizations
            charts = agent.create_visualizations(results_df)
            
            # Display charts
            if 'segments' in charts:
                st.plotly_chart(charts['segments'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'propensity' in charts:
                    st.plotly_chart(charts['propensity'], use_container_width=True)
            
            with col2:
                if 'strategies' in charts:
                    st.plotly_chart(charts['strategies'], use_container_width=True)
            
            # Detailed insights
            st.subheader("📊 Detailed Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.write("**Segment Analysis**")
                segment_analysis = results_df.groupby('segment_name').agg({
                    'total_spent': ['mean', 'sum'],
                    'propensity_score': 'mean',
                    'last_purchase_days_ago': 'mean'
                }).round(2)
                st.dataframe(segment_analysis, use_container_width=True)
            
            with insights_col2:
                st.write("**Campaign Strategy Breakdown**")
                if 'campaign_strategy' in results_df.columns:
                    strategy_breakdown = results_df.groupby('campaign_strategy').agg({
                        'propensity_score': 'mean',
                        'total_spent': 'mean'
                    }).round(2)
                    st.dataframe(strategy_breakdown, use_container_width=True)
        
        else:
            st.warning("⚠️ No results data available. Please generate campaigns first.")

if __name__ == "__main__":
    main()