import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import random
import json
from typing import Dict, List

# Configure page
st.set_page_config(
    page_title="MLOps Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
}

.alert-critical {
    background-color: #ffe6e6;
    border-left: 4px solid #ff4444;
    padding: 1rem;
    border-radius: 5px;
    margin: 0.5rem 0;
}

.alert-warning {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 1rem;
    border-radius: 5px;
    margin: 0.5rem 0;
}

.alert-info {
    background-color: #d1ecf1;
    border-left: 4px solid #17a2b8;
    padding: 1rem;
    border-radius: 5px;
    margin: 0.5rem 0;
}

.status-healthy { color: #28a745; font-weight: bold; }
.status-warning { color: #ffc107; font-weight: bold; }
.status-critical { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class MockDataGenerator:
    """Generate realistic mock data for demo purposes"""
    
    @staticmethod
    def generate_fraud_transaction():
        """Generate a mock fraud detection transaction"""
        merchants = ["Amazon", "Walmart", "Target", "Best Buy", "Shell", "McDonald's", "Starbucks"]
        categories = ["retail", "gas_station", "restaurant", "electronics", "grocery"]
        
        # Generate realistic patterns
        hour = random.randint(0, 23)
        is_night = hour < 6 or hour > 22
        is_weekend = random.choice([True, False])
        
        # Higher fraud probability for suspicious patterns
        base_fraud_prob = 0.05
        if is_night:
            base_fraud_prob += 0.3
        if hour in [2, 3, 4]:  # Very late night
            base_fraud_prob += 0.4
            
        amount = random.uniform(10, 5000)
        if amount > 2000:
            base_fraud_prob += 0.2
            
        location_risk = random.uniform(0, 1)
        if location_risk > 0.7:
            base_fraud_prob += 0.3
            
        fraud_probability = min(base_fraud_prob + random.uniform(-0.1, 0.1), 0.95)
        
        return {
            "transaction_id": f"TXN_{random.randint(100000, 999999)}",
            "user_id": f"USER_{random.randint(1000, 9999)}",
            "amount": round(amount, 2),
            "merchant": random.choice(merchants),
            "category": random.choice(categories),
            "location_risk_score": round(location_risk, 2),
            "hour_of_day": hour,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "fraud_probability": round(fraud_probability, 4),
            "is_fraud": fraud_probability > 0.5,
            "processing_time_ms": round(random.uniform(50, 200), 2),
            "timestamp": datetime.now() - timedelta(seconds=random.randint(0, 3600))
        }
    
    @staticmethod
    def generate_customer_service_query():
        """Generate a mock customer service interaction"""
        queries = [
            "I can't login to my account",
            "My order hasn't arrived yet",
            "I want to return this product",
            "The app keeps crashing on my phone",
            "I was charged twice for the same order",
            "How do I update my payment method?",
            "Can you help me track my package?",
            "I need to cancel my subscription",
            "The product I received is damaged",
            "I forgot my password"
        ]
        
        responses = [
            "I understand you're having trouble logging in. Let me help you reset your password.",
            "I apologize for the delay with your order. Let me check the status for you.",
            "I'd be happy to help you with your return. Our return policy allows returns within 30 days.",
            "I'm sorry to hear about the app crashes. Let's troubleshoot this together.",
            "I sincerely apologize for the double charge. I'll investigate this immediately.",
            "I can help you update your payment method. Please go to Account Settings.",
            "I'll be happy to help you track your package. Could you provide your order number?",
            "I can help you cancel your subscription. Let me process that for you.",
            "I'm sorry the product arrived damaged. I'll arrange a replacement immediately.",
            "I can help you reset your password. Please check your email for instructions."
        ]
        
        query_idx = random.randint(0, len(queries) - 1)
        response_time = random.uniform(0.5, 3.0)
        confidence = random.uniform(0.7, 0.98)
        
        return {
            "query": queries[query_idx],
            "response": responses[query_idx],
            "response_time": round(response_time, 2),
            "confidence_score": round(confidence, 3),
            "model_version": "customer-service-v2.1",
            "timestamp": datetime.now() - timedelta(seconds=random.randint(0, 1800)),
            "user_satisfied": random.choice([True, True, True, False])  # 75% satisfaction
        }

class MLOpsMetrics:
    """Generate MLOps monitoring metrics"""
    
    @staticmethod
    def get_system_metrics():
        """Get current system health metrics"""
        # Simulate realistic system metrics
        cpu_usage = random.uniform(20, 85)
        memory_usage = random.uniform(40, 90)
        disk_usage = random.uniform(10, 70)
        
        # Determine system status
        if cpu_usage > 80 or memory_usage > 85:
            status = "Critical"
        elif cpu_usage > 60 or memory_usage > 70:
            status = "Warning"
        else:
            status = "Healthy"
            
        return {
            "status": status,
            "cpu_usage": round(cpu_usage, 1),
            "memory_usage": round(memory_usage, 1),
            "disk_usage": round(disk_usage, 1),
            "uptime_hours": round(random.uniform(120, 720), 1),
            "requests_per_minute": random.randint(450, 1200),
            "error_rate": round(random.uniform(0.1, 2.5), 2)
        }
    
    @staticmethod
    def get_model_metrics():
        """Get model performance metrics"""
        fraud_accuracy = random.uniform(0.85, 0.96)
        cs_satisfaction = random.uniform(0.78, 0.94)
        
        return {
            "fraud_detection": {
                "accuracy": round(fraud_accuracy, 3),
                "precision": round(random.uniform(0.82, 0.95), 3),
                "recall": round(random.uniform(0.80, 0.93), 3),
                "f1_score": round(random.uniform(0.81, 0.94), 3),
                "auc_score": round(random.uniform(0.87, 0.97), 3),
                "data_drift_score": round(random.uniform(0.1, 0.6), 3),
                "model_version": "fraud-detector-v3.2",
                "last_retrained": datetime.now() - timedelta(days=random.randint(1, 7))
            },
            "customer_service": {
                "satisfaction_rate": round(cs_satisfaction, 3),
                "avg_response_time": round(random.uniform(1.2, 2.8), 2),
                "resolution_rate": round(random.uniform(0.85, 0.95), 3),
                "model_confidence": round(random.uniform(0.88, 0.96), 3),
                "model_version": "customer-service-v2.1",
                "last_updated": datetime.now() - timedelta(days=random.randint(1, 14))
            }
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 MLOps Platform Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    demo_type = st.sidebar.selectbox(
        "Choose Demo",
        ["Dashboard Overview", "Fraud Detection System", "Customer Service AI", "System Monitoring"]
    )
    
    # Initialize session state
    if 'fraud_predictions' not in st.session_state:
        st.session_state.fraud_predictions = []
    if 'cs_interactions' not in st.session_state:
        st.session_state.cs_interactions = []
    
    if demo_type == "Dashboard Overview":
        show_dashboard_overview()
    elif demo_type == "Fraud Detection System":
        show_fraud_detection_demo()
    elif demo_type == "Customer Service AI":
        show_customer_service_demo()
    elif demo_type == "System Monitoring":
        show_monitoring_demo()

def show_dashboard_overview():
    """Show high-level dashboard overview"""
    st.header("📊 Executive Dashboard")
    
    # Get current metrics
    system_metrics = MLOpsMetrics.get_system_metrics()
    model_metrics = MLOpsMetrics.get_model_metrics()
    
    # System Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if system_metrics["status"] == "Healthy" else "🟡" if system_metrics["status"] == "Warning" else "🔴"
        st.metric("System Status", f"{status_color} {system_metrics['status']}")
    
    with col2:
        st.metric("Requests/Min", f"{system_metrics['requests_per_minute']:,}")
    
    with col3:
        st.metric("Error Rate", f"{system_metrics['error_rate']}%")
    
    with col4:
        st.metric("Uptime", f"{system_metrics['uptime_hours']:.1f}h")
    
    # Model Performance Summary
    st.subheader("🎯 Model Performance Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Fraud Detection Model")
        fraud_metrics = model_metrics["fraud_detection"]
        st.metric("Accuracy", f"{fraud_metrics['accuracy']:.1%}")
        st.metric("AUC Score", f"{fraud_metrics['auc_score']:.3f}")
        st.metric("Data Drift", f"{fraud_metrics['data_drift_score']:.3f}")
        
        # Drift status
        drift_score = fraud_metrics['data_drift_score']
        if drift_score > 0.5:
            st.error("⚠️ High data drift detected - Retraining recommended")
        elif drift_score > 0.3:
            st.warning("⚡ Moderate data drift - Monitor closely")
        else:
            st.success("✅ Data drift within normal range")
    
    with col2:
        st.markdown("### Customer Service AI")
        cs_metrics = model_metrics["customer_service"]
        st.metric("Satisfaction Rate", f"{cs_metrics['satisfaction_rate']:.1%}")
        st.metric("Avg Response Time", f"{cs_metrics['avg_response_time']:.1f}s")
        st.metric("Resolution Rate", f"{cs_metrics['resolution_rate']:.1%}")
        
        # Performance status
        if cs_metrics['satisfaction_rate'] > 0.9:
            st.success("✅ Excellent customer satisfaction")
        elif cs_metrics['satisfaction_rate'] > 0.8:
            st.info("📈 Good performance, room for improvement")
        else:
            st.warning("⚠️ Performance below target")
    
    # Recent Alerts
    st.subheader("🚨 Recent Alerts")
    
    # Generate mock alerts
    alerts = [
        {"type": "INFO", "message": "Fraud detection model retrained successfully", "time": "2 hours ago"},
        {"type": "WARNING", "message": "Customer service response time increased by 15%", "time": "4 hours ago"},
        {"type": "CRITICAL", "message": "Memory usage exceeded 90% threshold", "time": "6 hours ago", "resolved": True}
    ]
    
    for alert in alerts:
        if alert["type"] == "CRITICAL":
            style_class = "alert-critical"
            icon = "🔴"
        elif alert["type"] == "WARNING":
            style_class = "alert-warning"
            icon = "🟡"
        else:
            style_class = "alert-info"
            icon = "🔵"
        
        resolved_text = " (RESOLVED)" if alert.get("resolved") else ""
        st.markdown(f"""
        <div class="{style_class}">
            {icon} <strong>{alert['type']}</strong>: {alert['message']}{resolved_text}
            <br><small>📅 {alert['time']}</small>
        </div>
        """, unsafe_allow_html=True)

def show_fraud_detection_demo():
    """Show fraud detection system demo"""
    st.header("🛡️ Real-Time Fraud Detection System")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Transaction Input")
        
        # Auto-generate transaction button
        if st.button("🎲 Generate Sample Transaction", type="primary"):
            transaction = MockDataGenerator.generate_fraud_transaction()
            
            # Store in session state
            st.session_state.fraud_predictions.append(transaction)
            
            # Show transaction details
            st.json(transaction)
            
            # Show prediction result
            risk_level = "HIGH" if transaction["fraud_probability"] > 0.7 else "MEDIUM" if transaction["fraud_probability"] > 0.3 else "LOW"
            
            if transaction["is_fraud"]:
                st.error(f"🚨 **FRAUD DETECTED** - Risk Level: {risk_level}")
                st.error(f"Fraud Probability: {transaction['fraud_probability']:.1%}")
            else:
                st.success(f"✅ **LEGITIMATE** - Risk Level: {risk_level}")
                st.info(f"Fraud Probability: {transaction['fraud_probability']:.1%}")
            
            st.info(f"⏱️ Processing Time: {transaction['processing_time_ms']}ms")
    
    with col2:
        st.subheader("Live Fraud Statistics")
        
        if st.session_state.fraud_predictions:
            df = pd.DataFrame(st.session_state.fraud_predictions)
            
            # Current metrics
            total_transactions = len(df)
            fraud_count = df['is_fraud'].sum()
            fraud_rate = fraud_count / total_transactions * 100
            avg_amount = df['amount'].mean()
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Transactions", total_transactions)
                st.metric("Fraud Detected", fraud_count)
            with col_b:
                st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
                st.metric("Avg Amount", f"${avg_amount:.2f}")
            
            # Fraud probability distribution
            fig = px.histogram(df, x='fraud_probability', nbins=20, 
                             title="Fraud Probability Distribution")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent transactions table
            st.subheader("Recent Transactions")
            recent_df = df.tail(5)[['transaction_id', 'amount', 'fraud_probability', 'is_fraud', 'processing_time_ms']]
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("Generate some transactions to see statistics")
    
    # Model Performance Section
    st.subheader("📈 Model Performance Metrics")
    
    model_metrics = MLOpsMetrics.get_model_metrics()["fraud_detection"]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{model_metrics['accuracy']:.1%}")
    with col2:
        st.metric("Precision", f"{model_metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{model_metrics['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{model_metrics['f1_score']:.3f}")

def show_customer_service_demo():
    """Show customer service AI demo"""
    st.header("💬 Customer Service AI Assistant")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Customer Query")
        
        # Manual input
        custom_query = st.text_input("Enter your question:", placeholder="e.g., I can't login to my account")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🎲 Generate Sample Query", type="secondary"):
                interaction = MockDataGenerator.generate_customer_service_query()
                st.session_state.cs_interactions.append(interaction)
                
                st.text_area("Customer Query:", value=interaction["query"], height=100, disabled=True)
                st.text_area("AI Response:", value=interaction["response"], height=150, disabled=True)
                
                col_x, col_y = st.columns(2)
                with col_x:
                    st.metric("Response Time", f"{interaction['response_time']}s")
                    st.metric("Confidence", f"{interaction['confidence_score']:.1%}")
                with col_y:
                    satisfaction_icon = "😊" if interaction["user_satisfied"] else "😞"
                    st.metric("User Satisfied", satisfaction_icon)
                    st.metric("Model Version", interaction["model_version"])
        
        with col_b:
            if custom_query and st.button("🚀 Get AI Response", type="primary"):
                # Simulate AI response
                with st.spinner("AI is thinking..."):
                    time.sleep(1)  # Simulate processing time
                
                # Generate mock response
                responses = {
                    "login": "I understand you're having trouble logging in. Let me help you reset your password by sending a reset link to your email.",
                    "order": "I apologize for the delay with your order. Let me check the status and provide you with an update.",
                    "return": "I'd be happy to help you with your return. Our return policy allows returns within 30 days of purchase.",
                    "app": "I'm sorry to hear about the app issues. Let's troubleshoot this together by first trying to restart the app.",
                    "charge": "I sincerely apologize for any billing issues. I'll investigate this immediately and ensure any errors are corrected."
                }
                
                # Simple keyword matching for demo
                response = "I understand your concern and I'm here to help. Let me assist you with this issue right away."
                for keyword, resp in responses.items():
                    if keyword in custom_query.lower():
                        response = resp
                        break
                
                st.text_area("AI Response:", value=response, height=150, disabled=True)
                
                # Mock metrics
                response_time = random.uniform(0.8, 2.5)
                confidence = random.uniform(0.85, 0.98)
                
                col_x, col_y = st.columns(2)
                with col_x:
                    st.metric("Response Time", f"{response_time:.1f}s")
                    st.metric("Confidence", f"{confidence:.1%}")
                with col_y:
                    st.info("Was this response helpful?")
                    col_thumb1, col_thumb2 = st.columns(2)
                    with col_thumb1:
                        st.button("👍 Yes")
                    with col_thumb2:
                        st.button("👎 No")
    
    with col2:
        st.subheader("Performance Analytics")
        
        if st.session_state.cs_interactions:
            df = pd.DataFrame(st.session_state.cs_interactions)
            
            # Performance metrics
            avg_response_time = df['response_time'].mean()
            avg_confidence = df['confidence_score'].mean()
            satisfaction_rate = df['user_satisfied'].mean()
            total_interactions = len(df)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Interactions", total_interactions)
                st.metric("Avg Response Time", f"{avg_response_time:.1f}s")
            with col_b:
                st.metric("Satisfaction Rate", f"{satisfaction_rate:.1%}")
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Response time chart
            fig = px.line(df.reset_index(), x='index', y='response_time', 
                         title="Response Time Trend")
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent interactions
            st.subheader("Recent Interactions")
            recent_df = df.tail(3)[['query', 'response_time', 'confidence_score', 'user_satisfied']]
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("Generate some interactions to see analytics")

def show_monitoring_demo():
    """Show system monitoring dashboard"""
    st.header("📊 System Monitoring & Alerts")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 5 seconds)")
    
    if auto_refresh:
        # Auto-refresh placeholder
        placeholder = st.empty()
        
        for i in range(12):  # Run for 1 minute
            with placeholder.container():
                display_monitoring_dashboard()
            time.sleep(5)
    else:
        # Manual refresh
        if st.button("🔄 Refresh Metrics"):
            display_monitoring_dashboard()
        else:
            display_monitoring_dashboard()

def display_monitoring_dashboard():
    """Display the monitoring dashboard content"""
    system_metrics = MLOpsMetrics.get_system_metrics()
    model_metrics = MLOpsMetrics.get_model_metrics()
    
    # System Health Overview
    st.subheader("🖥️ System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_color = "normal" if system_metrics["cpu_usage"] < 70 else "inverse"
        st.metric("CPU Usage", f"{system_metrics['cpu_usage']}%", delta=f"{random.uniform(-5, 5):.1f}%")
    
    with col2:
        memory_color = "normal" if system_metrics["memory_usage"] < 80 else "inverse"
        st.metric("Memory Usage", f"{system_metrics['memory_usage']}%", delta=f"{random.uniform(-3, 7):.1f}%")
    
    with col3:
        st.metric("Disk Usage", f"{system_metrics['disk_usage']}%", delta=f"{random.uniform(-1, 2):.1f}%")
    
    with col4:
        st.metric("Error Rate", f"{system_metrics['error_rate']}%", delta=f"{random.uniform(-0.5, 0.5):.2f}%")
    
    # Performance Charts
    st.subheader("📈 Performance Trends")
    
    # Generate time series data
    times = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU usage over time
        cpu_data = [random.uniform(30, 85) for _ in times]
        fig_cpu = go.Figure()
        fig_cpu.add_trace(go.Scatter(x=times, y=cpu_data, mode='lines', name='CPU Usage'))
        fig_cpu.update_layout(title="CPU Usage (24h)", yaxis_title="Usage %", height=300)
        st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        # Request rate over time
        request_data = [random.randint(400, 1200) for _ in times]
        fig_req = go.Figure()
        fig_req.add_trace(go.Scatter(x=times, y=request_data, mode='lines', name='Requests/min', line=dict(color='green')))
        fig_req.update_layout(title="Request Rate (24h)", yaxis_title="Requests/min", height=300)
        st.plotly_chart(fig_req, use_container_width=True)
    
    # Model Drift Monitoring
    st.subheader("🎯 Model Drift Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        drift_score = model_metrics["fraud_detection"]["data_drift_score"]
        
        # Create drift gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = drift_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Model Drift Score"},
            delta = {'reference': 0.3},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.5], 'color': "yellow"},
                    {'range': [0.5, 1], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5}}))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Alert summary
        st.markdown("### 🚨 Active Alerts")
        
        # Generate sample alerts based on metrics
        alerts = []
        
        if system_metrics["cpu_usage"] > 80:
            alerts.append({"level": "CRITICAL", "message": f"CPU usage at {system_metrics['cpu_usage']}%"})
        elif system_metrics["cpu_usage"] > 70:
            alerts.append({"level": "WARNING", "message": f"CPU usage elevated: {system_metrics['cpu_usage']}%"})
        
        if system_metrics["memory_usage"] > 85:
            alerts.append({"level": "CRITICAL", "message": f"Memory usage at {system_metrics['memory_usage']}%"})
        
        if drift_score > 0.5:
            alerts.append({"level": "WARNING", "message": f"High data drift detected: {drift_score:.3f}"})
        
        if system_metrics["error_rate"] > 2.0:
            alerts.append({"level": "CRITICAL", "message": f"Error rate elevated: {system_metrics['error_rate']}%"})
        
        if not alerts:
            st.success("✅ No active alerts - All systems healthy")
        else:
            for alert in alerts:
                if alert["level"] == "CRITICAL":
                    st.error(f"🔴 **{alert['level']}**: {alert['message']}")
                else:
                    st.warning(f"🟡 **{alert['level']}**: {alert['message']}")
    
    # Recent Activity Log
    st.subheader("📋 Recent Activity")
    
    activities = [
        f"[{datetime.now().strftime('%H:%M')}] Fraud model processed 1,247 transactions",
        f"[{(datetime.now() - timedelta(minutes=5)).strftime('%H:%M')}] Customer service AI handled 89 queries",
        f"[{(datetime.now() - timedelta(minutes=10)).strftime('%H:%M')}] System health check completed",
        f"[{(datetime.now() - timedelta(minutes=15)).strftime('%H:%M')}] Model drift check: Fraud detector - Score: {drift_score:.3f}",
        f"[{(datetime.now() - timedelta(minutes=20)).strftime('%H:%M')}] Auto-scaling triggered: Added 2 instances",
    ]
    
    for activity in activities:
        st.text(activity)

if __name__ == "__main__":
    main()
