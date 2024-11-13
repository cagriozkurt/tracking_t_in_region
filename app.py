import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import splrep, splev

st.title("Mouse Track Visualization and Time in Region Calculation")

# Step 1: Upload the ACT File
st.sidebar.header("Upload your .act file")
uploaded_file = st.sidebar.file_uploader("Choose a .act file", type=["act"])

# Global variables for track plotting and region calculation
x_min = st.sidebar.number_input("X Min", value=4.0, step=0.1)
x_max = st.sidebar.number_input("X Max", value=9.0, step=0.1)
y_min = st.sidebar.number_input("Y Min", value=4.0, step=0.1)
y_max = st.sidebar.number_input("Y Max", value=9.0, step=0.1)


# Function to parse .act file and extract the Data section
@st.cache_data
def parse_act_file(file_content):
    lines = file_content.splitlines()
    data_start_idx = lines.index("[Data]") + 1  # Find the start of the data
    
    # Extract raw data lines (semicolon-delimited)
    data_lines = lines[data_start_idx:]
    
    # Define columns: time (ms), x, y, z
    data = []
    for line in data_lines:
        line = line.replace(",", ".")  # Replace commas with dots for decimal points
        fields = line.split(";")  # Split by semicolon
        if len(fields) >= 4:
            data.append([float(fields[0]), float(fields[1]), float(fields[2]), float(fields[3])])
    
    # Convert the parsed data into a pandas DataFrame
    df = pd.DataFrame(data, columns=["t", "x", "y", "z"])
    return df

if uploaded_file is not None:
    # Step 2: Read and Parse the ACT File, Extract Data Section
    file_content = uploaded_file.read().decode('utf-8')  # Convert bytes to string
    df = parse_act_file(file_content)
    
    st.write("### Uploaded Data Preview")
    st.write(df.head())
    
    # Step 3: Function to Plot the Mouse Tracks with Boundary Box
    def plot_tracks(df, x_min, x_max, y_min, y_max):
        sns.set_palette("deep")
        plt.figure(figsize=(8, 8))
        plt.plot(df["x"], df["y"], linestyle="--", color="#4E79A7", alpha=0.5, linewidth=0.5, label="Mouse Path")
        plt.scatter(df["x"].iloc[0], df["y"].iloc[0], color="#59A14F", s=50, edgecolor="black", zorder=5, label="Start")
        plt.scatter(df["x"].iloc[-1], df["y"].iloc[-1], color="#F28E2B", s=50, edgecolor="black", zorder=5, label="End")
        plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color="#E15759", linestyle="--", linewidth=2, label="Boundary Box")
        plt.title("Mouse Track with Boundary Box", fontsize=16)
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.xlim([min(df["x"].min() - 1, x_min - 1), max(df["x"].max() + 1, x_max + 1)])
        plt.ylim([min(df["y"].min() - 1, y_min - 1), max(df["y"].max() + 1, y_max + 1)])
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(loc="lower right", fontsize=10)
        st.pyplot(plt)

    # Step 4: Display the Track Plot with Specified Region
    st.write("### Track Visualization")
    plot_tracks(df, x_min, x_max, y_min, y_max)

    # Step 5: Function to Calculate Time in Region
    def calculate_time_in_region(df, x_min, x_max, y_min, y_max):
        t_values = df["t"].values
        x_values = df["x"].values
        y_values = df["y"].values
        spline_x = splrep(t_values, x_values, s=0)
        spline_y = splrep(t_values, y_values, s=0)
        
        time_in_region_spline = 0

        for i in range(1, len(t_values)):
            previous_t = t_values[i - 1]
            current_t = t_values[i]
            time_interval = current_t - previous_t
            
            previous_x = splev(previous_t, spline_x)
            previous_y = splev(previous_t, spline_y)
            current_x = splev(current_t, spline_x)
            current_y = splev(current_t, spline_y)

            prev_inside = (x_min <= previous_x <= x_max) and (y_min <= previous_y <= y_max)
            curr_inside = (x_min <= current_x <= x_max) and (y_min <= current_y <= y_max)

            if prev_inside and curr_inside:
                time_in_region_spline += time_interval
            elif prev_inside or curr_inside:
                # Fine-grain sampling for boundary transitions
                fine_t = np.linspace(previous_t, current_t, num=10)
                for j in range(1, len(fine_t)):
                    fine_previous_t = fine_t[j - 1]
                    fine_current_t = fine_t[j]
                    fine_previous_x = splev(fine_previous_t, spline_x)
                    fine_previous_y = splev(fine_previous_t, spline_y)
                    fine_current_x = splev(fine_current_t, spline_x)
                    fine_current_y = splev(fine_current_t, spline_y)
                    
                    fine_prev_inside = (x_min <= fine_previous_x <= x_max) and (y_min <= fine_previous_y <= y_max)
                    fine_curr_inside = (x_min <= fine_current_x <= x_max) and (y_min <= fine_current_y <= y_max)
                    
                    if fine_prev_inside and fine_curr_inside:
                        time_in_region_spline += fine_current_t - fine_previous_t
        
        return time_in_region_spline / 1000.0  # Convert to seconds

    # Step 6: Calculate Time in Region
    st.write("### Time Spent in Specified Region")
    time_spent = calculate_time_in_region(df, x_min, x_max, y_min, y_max)
    st.write(f"Time in region: {time_spent:.3f} seconds")