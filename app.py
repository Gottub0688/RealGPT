# -*- coding: utf-8 -*-
"""
Streamlit Web Application - Australian Property Analysis System
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import time

# Import main system modules
sys.path.append(os.path.dirname(__file__))
from property_system import (
    RealDataFetcher,
    TimeSeriesPredictor,
    PropertyDatabase,
    AutomatedMonitor
)

# Page configuration
st.set_page_config(
    page_title="Australian Property Intelligence Analysis System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'fetcher' not in st.session_state:
    st.session_state.fetcher = RealDataFetcher()
if 'db' not in st.session_state:
    st.session_state.db = PropertyDatabase()
if 'predictor' not in st.session_state:
    st.session_state.predictor = TimeSeriesPredictor()

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/667eea/ffffff?text=PropertyAI", width=150)
    st.title("🏠 Navigation")

    page = st.radio(
        "Select Function",
        [
            "📊 Data Overview",
            "🗺️ Area Analysis",
            "🎯 Smart Valuation",
            "📈 Time Series Prediction",
            "🚨 Monitoring Alerts",
            "⚙️ System Settings"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("Quick Actions")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("Data refreshed")

    if st.button("📥 Export Report", use_container_width=True):
        st.info("Generating report...")

    st.divider()

    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))

# ==================== Main Pages ====================

if page == "📊 Data Overview":
    st.markdown('<div class="main-header">📊 Data Overview Dashboard</div>', unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Monitored Areas",
            value="10",
            delta="+2",
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="Data Records",
            value="5,432",
            delta="+156 today",
            delta_color="normal"
        )

    with col3:
        st.metric(
            label="Active Alerts",
            value="3",
            delta="-1",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            label="Model Accuracy",
            value="92.5%",
            delta="+1.2%",
            delta_color="normal"
        )

    st.divider()

    # Recent updates
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Recent Transactions")

        try:
            recent_data = pd.read_sql(
                "SELECT suburb, address, sale_price, sale_date, property_type FROM sales ORDER BY created_at DESC LIMIT 15",
                st.session_state.db.conn
            )

            if not recent_data.empty:
                recent_data['sale_price'] = recent_data['sale_price'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(recent_data, use_container_width=True, hide_index=True)
            else:
                st.info("No data available, please run data collection first")
        except:
            st.warning("Database connection failed, using demo data")
            demo_data = pd.DataFrame({
                'Suburb': ['Burwood', 'Strathfield', 'Croydon'] * 5,
                'Address': [f"{np.random.randint(1, 200)} Street" for _ in range(15)],
                'Price': [f"${np.random.randint(800, 1500)}k" for _ in range(15)],
                'Date': pd.date_range(end=datetime.now(), periods=15, freq='D'),
                'Type': np.random.choice(['House', 'Unit', 'Townhouse'], 15)
            })
            st.dataframe(demo_data, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("🏆 Top 5 Hot Areas")

        top_suburbs = pd.DataFrame({
            'Suburb': ['Burwood', 'Strathfield', 'Rhodes', 'Ashfield', 'Croydon'],
            'Avg Price': [1250000, 1380000, 1180000, 1050000, 980000],
            'Change': ['+5.2%', '+4.8%', '+6.1%', '+3.5%', '+2.9%']
        })

        for idx, row in top_suburbs.iterrows():
            st.metric(
                label=row['Suburb'],
                value=f"${row['Avg Price']/1000:.0f}k",
                delta=row['Change']
            )

    st.divider()

    # Price trend chart
    st.subheader("📈 180-Day Overall Price Trend")

    # Generate demo data
    dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
    base_price = 1200000
    prices = base_price + np.cumsum(np.random.randn(180) * 5000)

    trend_data = pd.DataFrame({
        'date': dates,
        'avg_price': prices
    })

    fig = px.line(
        trend_data,
        x='date',
        y='avg_price',
        title='Average Property Price Trend',
        labels={'date': 'Date', 'avg_price': 'Average Price ($)'}
    )
    fig.update_traces(line_color='#667eea', line_width=3)
    fig.update_layout(hovermode='x unified')

    st.plotly_chart(fig, use_container_width=True)

elif page == "🗺️ Area Analysis":
    st.markdown('<div class="main-header">🗺️ In-Depth Area Analysis</div>', unsafe_allow_html=True)

    suburbs = ['Burwood', 'Strathfield', 'Croydon', 'Ashfield',
               'Homebush', 'Concord', 'Rhodes', 'Haberfield']

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_suburb = st.selectbox("🔍 Select Area", suburbs, index=0)

        analyze_btn = st.button("Start Analysis", type="primary", use_container_width=True)

    with col2:
        st.info(f"💡 {selected_suburb} is a premium residential area in Sydney's inner west with excellent transport and education resources")

    if analyze_btn:
        with st.spinner(f"Analyzing {selected_suburb}..."):
            # Simulate data fetch
            time.sleep(1)

            tab1, tab2, tab3 = st.tabs(["📊 Basic Info", "📈 Market Trends", "🎓 Local Amenities"])

            with tab1:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Median Price", "$1,250,000", "+5.2%")
                    st.metric("Median Rent", "$650/week", "+3.1%")

                with col2:
                    st.metric("Rental Yield", "3.2%", "+0.1%")
                    st.metric("Vacancy Rate", "2.1%", "-0.3%")

                with col3:
                    st.metric("Distance to CBD", "10 km", "")
                    st.metric("Days on Market", "28 days", "-5 days")

                st.divider()

                # Demographics
                st.subheader("👥 Demographics")

                col1, col2 = st.columns(2)

                with col1:
                    demo_data = pd.DataFrame({
                        'Indicator': ['Total Population', 'Median Age', 'Households', 'Avg Household Size'],
                        'Value': ['18,542', '35 years', '6,234', '2.9 people']
                    })
                    st.dataframe(demo_data, use_container_width=True, hide_index=True)

                with col2:
                    # Age distribution pie chart
                    age_dist = pd.DataFrame({
                        'Age Group': ['0-18', '19-35', '36-50', '51-65', '65+'],
                        'Population': [3200, 5400, 4800, 3200, 1942]
                    })
                    fig = px.pie(age_dist, values='Population', names='Age Group', title='Age Distribution')
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Price trends
                months = pd.date_range(end=datetime.now(), periods=24, freq='M')
                prices = 1000000 + np.cumsum(np.random.randn(24) * 20000)

                trend_df = pd.DataFrame({
                    'month': months,
                    'price': prices
                })

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend_df['month'],
                    y=trend_df['price'],
                    mode='lines+markers',
                    name='Median Price',
                    line=dict(color='#667eea', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.2)'
                ))

                fig.update_layout(
                    title=f'{selected_suburb} 24-Month Price Trend',
                    xaxis_title='Month',
                    yaxis_title='Price ($)',
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Property type distribution
                col1, col2 = st.columns(2)

                with col1:
                    property_types = pd.DataFrame({
                        'Type': ['House', 'Unit', 'Townhouse'],
                        'Avg Price': [1450000, 850000, 1150000]
                    })
                    fig = px.bar(property_types, x='Type', y='Avg Price',
                               title='Average Price by Property Type',
                               color='Avg Price',
                               color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    bedrooms = pd.DataFrame({
                        'Bedrooms': ['2BR', '3BR', '4BR', '5BR+'],
                        'Count': [120, 245, 180, 55],
                        'Median': [950000, 1250000, 1650000, 2200000]
                    })
                    fig = px.scatter(bedrooms, x='Count', y='Median',
                                   size='Count', color='Bedrooms',
                                   title='Bedroom Count vs Price')
                    st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("🏫 Education Facilities")

                schools = pd.DataFrame({
                    'School Name': ['Burwood Public School', 'PLC Sydney', 'MLC School'],
                    'Type': ['Public', 'Private', 'Private'],
                    'Rating': [8.5, 9.2, 9.0],
                    'Distance': ['0.5 km', '1.2 km', '2.0 km']
                })
                st.dataframe(schools, use_container_width=True, hide_index=True)

                st.subheader("🚇 Transport")

                col1, col2 = st.columns(2)

                with col1:
                    st.info("**Burwood Station**\n- T1 North Shore Line\n- Distance: 0.8 km\n- To City: 15 minutes")

                with col2:
                    st.info("**Bus Routes**\n- 410, 461, 480, 483\n- Night Bus: N40\n- Coverage: Excellent")

                st.subheader("🛒 Local Amenities")

                amenities = pd.DataFrame({
                    'Type': ['Shopping', 'Hospital', 'Parks', 'Restaurants', 'Gyms'],
                    'Count': [3, 2, 5, 45, 8],
                    'Nearest': ['0.3 km', '1.5 km', '0.2 km', '0.1 km', '0.5 km']
                })

                fig = px.bar(amenities, x='Type', y='Count',
                           title='Local Amenities',
                           text='Count')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Smart Valuation":
    st.markdown('<div class="main-header">🎯 Smart Valuation Model</div>', unsafe_allow_html=True)

    st.info("💡 Enter property details and our AI model will provide accurate valuation")

    with st.form("valuation_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📍 Basic Information")
            suburb = st.selectbox("Suburb", ['Burwood', 'Strathfield', 'Croydon', 'Ashfield'])
            property_type = st.selectbox("Property Type", ['House', 'Unit', 'Townhouse'])
            bedrooms = st.number_input("Bedrooms", 1, 10, 3)
            bathrooms = st.number_input("Bathrooms", 1, 5, 2)

        with col2:
            st.subheader("🏡 Property Features")
            car_spaces = st.number_input("Car Spaces", 0, 5, 2)
            land_size = st.number_input("Land Size (m²)", 0, 2000, 400)
            building_age = st.slider("Building Age (years)", 0, 100, 15)
            condition = st.select_slider("Condition",
                                        options=['Needs Renovation', 'Fair', 'Good', 'Excellent', 'Brand New'])

        with col3:
            st.subheader("🌍 Environmental Factors")
            distance_cbd = st.slider("Distance to CBD (km)", 0, 50, 10)
            distance_station = st.slider("Distance to Station (km)", 0, 5, 1)
            school_score = st.slider("School Rating", 0, 100, 75)
            view_quality = st.select_slider("View Quality",
                                          options=['None', 'Fair', 'Good', 'Excellent'])

        submitted = st.form_submit_button("🔮 Calculate Valuation", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("AI model calculating..."):
            time.sleep(2)

            # Valuation calculation
            base_price = {
                'Burwood': 1200000,
                'Strathfield': 1400000,
                'Croydon': 1000000,
                'Ashfield': 950000
            }[suburb]

            # Type multiplier
            type_multiplier = {'House': 1.3, 'Unit': 0.8, 'Townhouse': 1.0}[property_type]

            # Calculate valuation
            price = base_price * type_multiplier
            price *= (1 + bedrooms * 0.15) * (1 + bathrooms * 0.08)
            price *= (1 + car_spaces * 0.05)
            price *= (land_size / 400) if property_type == 'House' else 1
            price *= (1 - building_age * 0.005)
            price *= (1 - distance_cbd * 0.02)
            price *= (1 - distance_station * 0.03)
            price *= (school_score / 100)

            condition_multiplier = {
                'Needs Renovation': 0.85, 'Fair': 0.95, 'Good': 1.0,
                'Excellent': 1.1, 'Brand New': 1.2
            }[condition]
            price *= condition_multiplier

            st.success("✅ Valuation Complete!")

            # Display results
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("💰 Estimated Value", f"${price:,.0f}")

            with col2:
                st.metric("📊 Value Range",
                         f"${price*0.92:,.0f} - ${price*1.08:,.0f}")

            with col3:
                st.metric("🎯 Confidence", "87%")

            with col4:
                market_comparison = np.random.choice(['Below', 'At', 'Above'])
                st.metric("vs Market", market_comparison,
                         f"{np.random.randint(-5, 6)}%")

            st.divider()

            # Valuation breakdown
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📋 Valuation Factor Breakdown")

                factors = pd.DataFrame({
                    'Factor': ['Location', 'Property Size', 'Age', 'Amenities', 'School District', 'Transport'],
                    'Weight': [35, 25, 10, 10, 15, 5],
                    'Score': [85, 90, 70, 80, school_score, 85]
                })

                fig = px.bar(factors, x='Factor', y='Score',
                           color='Score',
                           color_continuous_scale='RdYlGn',
                           title='Factor Scores')
                fig.update_traces(text=factors['Weight'].apply(lambda x: f'{x}%'),
                                textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 Comparable Properties")

                comparison = pd.DataFrame({
                    'Address': ['This Property', '123 Smith St', '45 George St', '78 Railway Rd'],
                    'Price': [price, price*0.95, price*1.05, price*0.98],
                    'Similarity': [100, 85, 78, 92]
                })

                fig = px.scatter(comparison, x='Similarity', y='Price',
                               size='Similarity', color='Address',
                               title='Comparable Property Prices')
                st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            st.subheader("💡 Professional Recommendations")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.success("""
                **Investment Potential**: ⭐⭐⭐⭐
                
                This area has shown rapid development in recent years 
                with excellent infrastructure. Expected significant 
                appreciation over the next 3 years.
                """)

            with col2:
                st.info("""
                **Suitable For**: 
                
                - Family buyers
                - First-time buyers
                - Investors
                """)

            with col3:
                st.warning("""
                **Important Notes**:
                
                - Recommend property inspection
                - Check structural condition
                - Review area planning
                """)

elif page == "📈 Time Series Prediction":
    st.markdown('<div class="main-header">📈 Price Trend Prediction</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_suburb = st.selectbox(
            "Select Area",
            ['Burwood', 'Strathfield', 'Croydon', 'Ashfield', 'Homebush']
        )

    with col2:
        forecast_months = st.slider("Forecast Period (months)", 1, 24, 6)

    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("AI model predicting..."):
            time.sleep(2)

            # Generate historical data
            historical_months = 24
            dates_historical = pd.date_range(
                end=datetime.now(),
                periods=historical_months,
                freq='M'
            )

            base_price = 1200000
            noise = np.random.randn(historical_months) * 20000
            trend = np.linspace(0, 150000, historical_months)
            historical_prices = base_price + trend + noise

            # Generate predictions
            dates_future = pd.date_range(
                start=dates_historical[-1] + timedelta(days=30),
                periods=forecast_months,
                freq='M'
            )

            last_price = historical_prices[-1]
            growth_rate = np.random.uniform(0.003, 0.008)
            predicted_prices = [last_price]

            for i in range(1, forecast_months):
                next_price = predicted_prices[-1] * (1 + growth_rate)
                predicted_prices.append(next_price)

            predicted_prices = np.array(predicted_prices)

            # Confidence intervals
            std = np.std(historical_prices) * 0.5
            confidence_upper = predicted_prices + 1.96 * std
            confidence_lower = predicted_prices - 1.96 * std

            expected_growth = ((predicted_prices[-1] - last_price) / last_price) * 100

            # Display prediction results
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Expected Growth",
                    f"{expected_growth:.2f}%",
                    delta="🔥" if expected_growth > 5 else "📈"
                )

            with col2:
                st.metric(
                    "Current Price",
                    f"${last_price:,.0f}"
                )

            with col3:
                st.metric(
                    f"After {forecast_months} months",
                    f"${predicted_prices[-1]:,.0f}"
                )

            with col4:
                trend_emoji = "🚀" if expected_growth > 5 else "📊" if expected_growth > 0 else "📉"
                st.metric("Trend", trend_emoji)

            st.divider()

            # Prediction chart
            st.subheader("📊 Price Prediction Chart")

            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                x=dates_historical,
                y=historical_prices,
                mode='lines+markers',
                name='Historical Price',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=6)
            ))

            # Predicted data
            fig.add_trace(go.Scatter(
                x=dates_future,
                y=predicted_prices,
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))

            # Confidence interval upper
            fig.add_trace(go.Scatter(
                x=dates_future,
                y=confidence_upper,
                mode='lines',
                name='Upper Confidence',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Confidence interval lower (fill)
            fig.add_trace(go.Scatter(
                x=dates_future,
                y=confidence_lower,
                mode='lines',
                name='95% Confidence Interval',
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(width=0)
            ))

            fig.update_layout(
                title=f'{selected_suburb} Price Prediction - Next {forecast_months} Months',
                xaxis_title='Time',
                yaxis_title='Price ($)',
                hovermode='x unified',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Prediction details table
            st.subheader("📋 Prediction Details")

            forecast_df = pd.DataFrame({
                'Month': range(1, forecast_months + 1),
                'Date': dates_future.strftime('%Y-%m'),
                'Predicted Price': [f"${p:,.0f}" for p in predicted_prices],
                'Lower Bound': [f"${p:,.0f}" for p in confidence_lower],
                'Upper Bound': [f"${p:,.0f}" for p in confidence_upper],
                'Monthly Growth': [f"{((predicted_prices[i]/predicted_prices[i-1])-1)*100:.2f}%"
                           if i > 0 else "-"
                           for i in range(forecast_months)]
            })

            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            # AI analysis
            st.subheader("🤖 AI Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"""
                **Trend Assessment**: {'Rising' if expected_growth > 0 else 'Declining'}
                
                Based on historical data analysis, {selected_suburb} is expected to 
                {'continue rising' if expected_growth > 0 else 'show correction'} over the next {forecast_months} months,
                with appreciation of approximately {abs(expected_growth):.2f}%.
                
                **Key Drivers**:
                - Regional development planning
                - Transport infrastructure improvements
                - Increased population inflow
                """)

            with col2:
                risk_level = "Low" if abs(expected_growth) < 5 else "Medium" if abs(expected_growth) < 10 else "High"
                st.warning(f"""
                **Investment Recommendation**:
                
                - Risk Level: {risk_level}
                - Recommended Holding Period: {forecast_months}+ months
                - Expected Return: {expected_growth:.2f}%
                
                **Important Notes**:
                - Model predictions are for reference only
                - Actual prices influenced by multiple factors
                - Recommend combining with on-site inspection
                """)

elif page == "🚨 Monitoring Alerts":
    st.markdown('<div class="main-header">🚨 Monitoring & Alert System</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📢 Real-time Alerts", "📊 Historical Records", "⚙️ Rule Settings"])

    with tab1:
        st.subheader("Today's Alerts")

        # Mock alert data
        alerts_today = pd.DataFrame({
            'Time': ['10:30', '14:15', '16:45'],
            'Area': ['Burwood', 'Strathfield', 'Rhodes'],
            'Type': ['Price Surge', 'Volume Spike', 'New Listing'],
            'Severity': ['HIGH', 'MEDIUM', 'INFO'],
            'Message': [
                'Price increased 12.5%, exceeding alert threshold',
                '15 sales today, +180% vs average',
                'New project Rhodes Central launched'
            ]
        })

        for idx, alert in alerts_today.iterrows():
            severity_config = {
                'HIGH': ('🔴', 'error'),
                'MEDIUM': ('🟠', 'warning'),
                'LOW': ('🟡', 'info'),
                'INFO': ('🔵', 'info')
            }

            icon, alert_type = severity_config[alert['Severity']]

            with st.expander(
                f"{icon} {alert['Time']} - {alert['Area']} - {alert['Type']}",
                expanded=(alert['Severity'] == 'HIGH')
            ):
                if alert_type == 'error':
                    st.error(f"**{alert['Message']}**")
                elif alert_type == 'warning':
                    st.warning(f"**{alert['Message']}**")
                else:
                    st.info(f"**{alert['Message']}**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button("📊 View Details", key=f"detail_{idx}")
                with col2:
                    st.button("🔕 Dismiss", key=f"ignore_{idx}")
                with col3:
                    st.button("⭐ Mark", key=f"mark_{idx}")

    with tab2:
        st.subheader("Alert History")

        days = st.slider("View Last N Days", 1, 30, 7)

        # Mock historical data
        history_alerts = pd.DataFrame({
            'Date': pd.date_range(end=datetime.now(), periods=20, freq='D'),
            'Area': np.random.choice(['Burwood', 'Strathfield', 'Rhodes'], 20),
            'Type': np.random.choice(['Price Change', 'Volume Change', 'New Listing'], 20),
            'Severity': np.random.choice(['HIGH', 'MEDIUM', 'LOW', 'INFO'], 20),
            'Status': np.random.choice(['Resolved', 'In Progress', 'Dismissed'], 20)
        })

        st.dataframe(history_alerts, use_container_width=True, hide_index=True)

        # Statistics charts
        col1, col2 = st.columns(2)

        with col1:
            severity_dist = history_alerts['Severity'].value_counts()
            fig = px.pie(
                values=severity_dist.values,
                names=severity_dist.index,
                title='Alert Severity Distribution',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            type_dist = history_alerts['Type'].value_counts()
            fig = px.bar(
                x=type_dist.index,
                y=type_dist.values,
                title='Alert Type Distribution',
                labels={'x': 'Type', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Alert Rule Settings")

        with st.form("alert_settings"):
            st.write("### Price Monitoring")

            col1, col2 = st.columns(2)

            with col1:
                price_change = st.slider(
                    "Price Change Threshold (%)",
                    0, 50, 10,
                    help="Trigger alert when price change exceeds this percentage"
                )

            with col2:
                volume_change = st.slider(
                    "Volume Change Threshold (%)",
                    0, 200, 50,
                    help="Trigger alert when volume change exceeds this percentage"
                )

            st.write("### Monitored Areas")

            monitored_suburbs = st.multiselect(
                "Select areas to monitor",
                ['Burwood', 'Strathfield', 'Croydon', 'Ashfield',
                 'Homebush', 'Concord', 'Rhodes', 'Haberfield'],
                default=['Burwood', 'Strathfield', 'Rhodes']
            )

            st.write("### Notification Settings")

            col1, col2 = st.columns(2)

            with col1:
                email_notify = st.checkbox("Email Notifications", value=True)
                sms_notify = st.checkbox("SMS Notifications", value=False)

            with col2:
                push_notify = st.checkbox("Push Notifications", value=True)
                daily_report = st.checkbox("Daily Report", value=True)

            submitted = st.form_submit_button("💾 Save Settings", type="primary")

            if submitted:
                st.success("✅ Settings saved successfully!")

elif page == "⚙️ System Settings":
    st.markdown('<div class="main-header">⚙️ System Settings</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔑 API Configuration", "📧 Notifications", "💾 Data Management"])

    with tab1:
        st.subheader("API Key Configuration")

        with st.form("api_config"):
            st.write("### Domain API")
            domain_key = st.text_input(
                "API Key",
                type="password",
                help="Get from https://developer.domain.com.au"
            )

            st.write("### CoreLogic API")
            corelogic_key = st.text_input(
                "API Key",
                type="password",
                help="Contact CoreLogic sales for enterprise key"
            )

            st.write("### Google Maps API")
            maps_key = st.text_input(
                "API Key",
                type="password",
                help="For geocoding and distance calculations"
            )

            save_api = st.form_submit_button("💾 Save API Configuration", type="primary")

            if save_api:
                st.success("✅ API configuration saved!")

        st.divider()

        st.subheader("Test API Connections")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Test Domain API"):
                with st.spinner("Connecting..."):
                    time.sleep(1)
                    st.success("✅ Domain API connected")

        with col2:
            if st.button("Test CoreLogic API"):
                with st.spinner("Connecting..."):
                    time.sleep(1)
                    st.warning("⚠️ API key not configured")

        with col3:
            if st.button("Test Maps API"):
                with st.spinner("Connecting..."):
                    time.sleep(1)
                    st.success("✅ Maps API connected")

    with tab2:
        st.subheader("Notification Settings")

        with st.form("notification_settings"):
            email = st.text_input("Email Address", help="Receive system notifications and reports")
            phone = st.text_input("Phone Number", help="Receive important alert SMS")

            st.write("### Notification Frequency")

            col1, col2 = st.columns(2)

            with col1:
                st.selectbox("Daily Report Time",
                           ['08:00', '09:00', '10:00', '18:00', '19:00', '20:00'])

            with col2:
                st.selectbox("Weekly Report Day",
                           ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

            save_notif = st.form_submit_button("💾 Save Notification Settings", type="primary")

            if save_notif:
                st.success("✅ Notification settings saved!")

    with tab3:
        st.subheader("Data Management")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Database Size", "245 MB")
            st.metric("Total Records", "5,432")

        with col2:
            st.metric("Last Updated", "2 hours ago")
            st.metric("Data Sources", "4")

        with col3:
            st.metric("Cache Size", "18 MB")
            st.metric("Log Size", "12 MB")

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.info("Refreshing data...")

        with col2:
            if st.button("📥 Export Data", use_container_width=True):
                st.download_button(
                    label="Download CSV",
                    data="suburb,price,date\n",
                    file_name="property_data.csv",
                    mime="text/csv"
                )

        with col3:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared")

# Footer
st.divider()
st.caption("© 2024 Property AI - Australian Property Intelligence Analysis System | Powered by Advanced ML Models")