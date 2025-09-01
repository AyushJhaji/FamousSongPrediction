import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Configure page
st.set_page_config(
    page_title="🎵 Song Popularity Predictor",
    page_icon="🎵",
    layout="wide"
)

# Data cleaning function
@st.cache_data
def clean_streams_data(df):
    """Clean corrupted streams data"""
    df = df.copy()
    
    # Convert streams to string first
    df['streams'] = df['streams'].astype(str)
    
    # Handle corrupted rows where multiple values are concatenated
    def extract_first_number(value):
        if pd.isna(value) or value == 'nan':
            return np.nan
        
        # Convert to string and extract first sequence of digits
        str_val = str(value).strip()
        
        # If it looks like concatenated data (very long string of digits)
        if len(str_val) > 15 and str_val.isdigit():
            # Take first reasonable number (8-12 digits for stream counts)
            for length in [10, 9, 8, 11, 12]:
                if len(str_val) >= length:
                    candidate = str_val[:length]
                    num = int(candidate)
                    # Reasonable stream count range
                    if 1000 <= num <= 10000000000:
                        return num
        
        # Try to extract first number from mixed content
        numbers = re.findall(r'\d+', str_val)
        if numbers:
            for num_str in numbers:
                if 4 <= len(num_str) <= 12:  # Reasonable stream count length
                    num = int(num_str)
                    if 1000 <= num <= 10000000000:
                        return num
        
        # If all else fails, try converting directly
        try:
            return float(str_val)
        except:
            return np.nan
    
    # Apply cleaning
    df['streams_clean'] = df['streams'].apply(extract_first_number)
    
    # Remove rows where we couldn't extract a reasonable stream count
    df = df.dropna(subset=['streams_clean'])
    df['streams'] = df['streams_clean'].astype(int)
    df = df.drop('streams_clean', axis=1)
    
    return df

# Load model and data
@st.cache_resource
def load_model():
    try:
        return joblib.load('song_pop_model.pkl')
    except FileNotFoundError:
        st.error("Model file 'song_pop_model.pkl' not found. Please train the model first.")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('spotify-2023.csv', encoding='latin1')
        # Clean the corrupted data
        df_clean = clean_streams_data(df)
        st.success(f"Data loaded and cleaned. {len(df_clean)} valid records found.")
        return df_clean
    except FileNotFoundError:
        st.error("Dataset 'spotify-2023.csv' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

model = load_model()
df = load_data()

# Main title
st.title('🎵 Song Popularity Predictor')
st.markdown("Predict song streaming success based on audio features and platform presence")

if model is not None and df is not None:
    # Show data quality info
    st.info(f"📊 Dataset contains {len(df):,} songs with valid stream data")
    
    # Sidebar for inputs
    st.sidebar.header('🎛️ Enter Song Features')
    
    # Input fields
    artist_count = st.sidebar.number_input('Artist Count', min_value=1, max_value=10, value=1)
    released_year = st.sidebar.number_input('Released Year', min_value=1900, max_value=2025, value=2023)
    released_month = st.sidebar.number_input('Released Month', min_value=1, max_value=12, value=6)
    released_day = st.sidebar.number_input('Released Day', min_value=1, max_value=31, value=15)
    
    st.sidebar.subheader('🎧 Audio Features')
    danceability = st.sidebar.slider('Danceability %', min_value=0, max_value=100, value=50)
    energy = st.sidebar.slider('Energy %', min_value=0, max_value=100, value=50)
    valence = st.sidebar.slider('Valence %', min_value=0, max_value=100, value=50)
    acousticness = st.sidebar.slider('Acousticness %', min_value=0, max_value=100, value=50)
    instrumentalness = st.sidebar.slider('Instrumentalness %', min_value=0, max_value=100, value=50)
    speechiness = st.sidebar.slider('Speechiness %', min_value=0, max_value=100, value=50)
    liveness = st.sidebar.slider('Liveness %', min_value=0, max_value=100, value=50)
    bpm = st.sidebar.number_input('BPM', min_value=50, max_value=200, value=120)
    
    st.sidebar.subheader('📱 Platform Presence')
    spotify_playlists = st.sidebar.number_input('Spotify Playlists', min_value=0, value=100)
    spotify_charts = st.sidebar.number_input('Spotify Charts', min_value=0, value=10)
    apple_playlists = st.sidebar.number_input('Apple Playlists', min_value=0, value=50)
    apple_charts = st.sidebar.number_input('Apple Charts', min_value=0, value=5)
    deezer_playlists = st.sidebar.number_input('Deezer Playlists', min_value=0, value=30)
    deezer_charts = st.sidebar.number_input('Deezer Charts', min_value=0, value=3)
    shazam_charts = st.sidebar.number_input('Shazam Charts', min_value=0, value=20)
    
    # Prediction button
    if st.sidebar.button('🎯 Predict Streams', type='primary'):
        try:
            # Create input dataframe with available features
            available_features = []
            input_dict = {}
            
            # Check which features exist in the original training data
            basic_features = {
                'artist_count': artist_count,
                'released_year': released_year,
                'danceability_%': danceability,
                'energy_%': energy,
                'valence_%': valence,
                'acousticness_%': acousticness,
                'instrumentalness_%': instrumentalness,
                'speechiness_%': speechiness,
                'liveness_%': liveness,
                'bpm': bpm
            }
            
            # Add features that exist in the dataset
            for feature, value in basic_features.items():
                if feature in df.columns:
                    input_dict[feature] = [value]
                    available_features.append(feature)
            
            # Add platform features if they exist
            platform_features = {
                'released_month': released_month,
                'released_day': released_day,
                'in_spotify_playlists': spotify_playlists,
                'in_spotify_charts': spotify_charts,
                'in_apple_playlists': apple_playlists,
                'in_apple_charts': apple_charts,
                'in_deezer_playlists': deezer_playlists,
                'in_deezer_charts': deezer_charts,
                'in_shazam_charts': shazam_charts
            }
            
            for feature, value in platform_features.items():
                if feature in df.columns:
                    input_dict[feature] = [value]
                    available_features.append(feature)
            
            input_data = pd.DataFrame(input_dict)
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success(f'🎉 Predicted Streams: **{prediction:,.0f}**')
            
            # Categorize success level
            if prediction > 500_000_000:
                st.balloons()
                st.write("🌟 **Viral Hit Potential!** This song could be a massive success!")
            elif prediction > 100_000_000:
                st.write("🔥 **High Success Potential!** This song has strong commercial appeal!")
            elif prediction > 50_000_000:
                st.write("✨ **Good Success Potential!** This song could perform well!")
            else:
                st.write("💡 **Modest Success Potential.** Consider optimizing features for better performance.")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("This might be due to feature mismatch between the model and input data.")
    
    # Dataset insights with cleaned data
    st.header('📈 Dataset Insights')
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Total Songs", f"{len(df):,}")
        if 'streams' in df.columns:
            try:
                avg_streams = df['streams'].mean()
                st.metric("Avg Streams", f"{avg_streams:,.0f}")
            except:
                st.metric("Avg Streams", "Data Error")
    
    with col4:
        if 'bpm' in df.columns:
            try:
                mode_bpm = df['bpm'].mode().iloc[0] if len(df['bpm'].mode()) > 0 else "N/A"
                st.metric("Most Common BPM", f"{mode_bpm}")
            except:
                st.metric("Most Common BPM", "N/A")
        
        if 'danceability_%' in df.columns:
            try:
                avg_dance = df['danceability_%'].mean()
                st.metric("Avg Danceability", f"{avg_dance:.1f}%")
            except:
                st.metric("Avg Danceability", "N/A")
    
    with col5:
        if 'released_year' in df.columns:
            try:
                mode_year = df['released_year'].mode().iloc[0] if len(df['released_year'].mode()) > 0 else "N/A"
                st.metric("Most Common Year", f"{mode_year}")
            except:
                st.metric("Most Common Year", "N/A")
        
        if 'energy_%' in df.columns:
            try:
                avg_energy = df['energy_%'].mean()
                st.metric("Avg Energy", f"{avg_energy:.1f}%")
            except:
                st.metric("Avg Energy", "N/A")
    
    # Show sample of cleaned data
    st.header('🔍 Data Preview')
    st.write("Sample of cleaned data:")
    display_cols = ['track_name', 'artist(s)_name', 'streams', 'danceability_%', 'energy_%', 'valence_%']
    available_display_cols = [col for col in display_cols if col in df.columns]
    if available_display_cols:
        st.dataframe(df[available_display_cols].head(10))
    
    # Marketing insights
    st.header('🎯 Success Strategies')
    st.markdown("""
    ### 🚀 Key Insights for Music Success:
    
    **🎵 Audio Features:**
    - **Danceability > 70%**: Creates engaging, shareable content
    - **Energy 60-80%**: Sweet spot for mainstream appeal
    - **Valence > 50%**: Positive songs perform better
    
    **📱 Platform Strategy:**
    - **Spotify Playlists**: Aim for 500+ for viral potential
    - **Cross-Platform**: Presence on multiple platforms amplifies reach
    - **Chart Performance**: Even small chart positions boost visibility
    
    **🤝 Collaboration:**
    - **Multiple Artists**: 2-3 artists often outperform solo tracks
    - **Release Timing**: Mid-year releases show strong performance
    """)

else:
    st.error("Please ensure both 'song_pop_model.pkl' and 'spotify-2023.csv' files are available.")
    st.info("The app detected data corruption issues. Please check your CSV file.")

# Run instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📝 **How to Run Properly:**
1. Use: `streamlit run app.py`
2. NOT: `python app.py`

### 🛠️ **If You See Errors:**
- Data corruption detected in streams column
- The app will attempt to clean the data automatically
- Some records may be dropped due to corruption
""")
