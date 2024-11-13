import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import splrep, splev

st.title("Mouse Track Visualization and Time in Region Calculation")

# Display GitHub repository link (Markdown format)
st.markdown(
    """
[View this project on GitHub](https://github.com/cagriozkurt/tracking_time_in_region)
"""
)


# Tabs for navigation
tabs = st.tabs(["Home", "About"])

# Global restrictions for x_min, x_max, y_min, y_max
MIN_BOUND = 1.0
MAX_BOUND = 15.0

# Global variables for track plotting and region calculation (used in Home page)
x_min = st.sidebar.number_input("X Min", value=5.0, min_value=MIN_BOUND, max_value=MAX_BOUND, step=0.1)
x_max = st.sidebar.number_input("X Max", value=10.0, min_value=MIN_BOUND, max_value=MAX_BOUND, step=0.1)
y_min = st.sidebar.number_input("Y Min", value=5.0, min_value=MIN_BOUND, max_value=MAX_BOUND, step=0.1)
y_max = st.sidebar.number_input("Y Max", value=10.0, min_value=MIN_BOUND, max_value=MAX_BOUND, step=0.1)

# Validation block to ensure x_min < x_max and y_min < y_max
if x_min > x_max:
    st.sidebar.error("Error: X Min cannot be greater than X Max.")
if y_min > y_max:
    st.sidebar.error("Error: Y Min cannot be greater than Y Max.")

# The "Home" Tab
with tabs[0]:
    st.header("Upload Your Mouse Tracking Data")

    # Step 1: Upload the ACT File
    st.sidebar.header("Upload your .act file")
    uploaded_file = st.sidebar.file_uploader("Choose a .act file", type=["act"])

    # Function to parse .act files and extract the data section
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
                try:
                    data.append(
                        [
                            float(fields[0]),
                            float(fields[1]),
                            float(fields[2]),
                            float(fields[3]),
                        ]
                    )
                except Exception as e:
                    st.warning(
                        f"Warning: Invalid data in line: {line}. Skipping this line."
                    )

        # Convert the parsed data into a pandas DataFrame
        df = pd.DataFrame(data, columns=["t", "x", "y", "z"])

        # Ensure we remove rows with NaN values and check for valid rows
        df.dropna(inplace=True)
        if df.empty:
            st.error("Error: The parsed data contains no valid points.")

        return df

    def check_for_gaps(df):
        t_values = df["t"].values

        # Check for time irregularities (gaps)
        time_diffs = np.diff(t_values)
        if np.any(time_diffs <= 0):
            st.error(
                "Warning: The time values are not strictly increasing or have duplicates."
            )
            return False

        # Check for large time gaps (e.g., 10x the smallest reading interval)
        reading_interval = df["t"].diff().min()
        large_gaps = time_diffs > reading_interval * 10
        if np.any(large_gaps):
            st.error(
                "Warning: Large gaps detected in the time values. Review your data."
            )
            return False

        # Check for missing coordinates (e.g., NaNs already dropped but validate)
        if df.isnull().values.any():
            st.error("Warning: Missing data detected in coordinates.")
            return False

        return True

    if uploaded_file is not None:
        # Step 2: Read and Parse the ACT File, Extract Data Section
        file_content = uploaded_file.read().decode("utf-8")  # Convert bytes to string
        df = parse_act_file(file_content)

        if not df.empty:
            st.write("### Uploaded Data Preview")
            st.write(df.head())

            # Step 3: Check for Gaps in Data
            if not check_for_gaps(df):
                st.error(
                    "Error: The uploaded data contains gaps. Please fix the data and re-upload."
                )
            else:

                # Step 3: Plotting Mouse Tracks with Boundary Box
                def plot_tracks(df, x_min, x_max, y_min, y_max):
                    sns.set_palette("deep")
                    plt.figure(figsize=(8, 8))
                    plt.plot(
                        df["x"],
                        df["y"],
                        linestyle="--",
                        color="#4E79A7",
                        alpha=0.5,
                        linewidth=0.5,
                        label="Mouse Path",
                    )
                    plt.scatter(
                        df["x"].iloc[0],
                        df["y"].iloc[0],
                        color="#59A14F",
                        s=50,
                        edgecolor="black",
                        zorder=5,
                        label="Start",
                    )
                    plt.scatter(
                        df["x"].iloc[-1],
                        df["y"].iloc[-1],
                        color="#F28E2B",
                        s=50,
                        edgecolor="black",
                        zorder=5,
                        label="End",
                    )
                    plt.plot(
                        [x_min, x_min, x_max, x_max, x_min],
                        [y_min, y_max, y_max, y_min, y_min],
                        color="#E15759",
                        linestyle="--",
                        linewidth=2,
                        label="Boundary Box",
                    )
                    st.write("### Track Visualization")
                    plt.title("Mouse Track with Boundary Box", fontsize=16)
                    plt.xlabel("X-coordinate")
                    plt.ylabel("Y-coordinate")
                    plt.xlim(
                        [
                            min(df["x"].min() - 1, x_min - 1),
                            max(df["x"].max() + 1, x_max + 1),
                        ]
                    )
                    plt.ylim(
                        [
                            min(df["y"].min() - 1, y_min - 1),
                            max(df["y"].max() + 1, y_max + 1),
                        ]
                    )
                    plt.grid(True, linestyle="--", alpha=0.3)
                    plt.legend(loc="lower right", fontsize=10)
                    st.pyplot(plt)

                # Display the track plot
                plot_tracks(df, x_min, x_max, y_min, y_max)

                # Step 4: Time in Region Calculation
                def calculate_time_in_region(df, x_min, x_max, y_min, y_max):
                    try:
                        t_values = df["t"].values
                        x_values = df["x"].values
                        y_values = df["y"].values

                        # Fit splines for x and y over time
                        spline_x = splrep(t_values, x_values, s=0)
                        spline_y = splrep(t_values, y_values, s=0)
                    except Exception as e:
                        st.error(f"Error: Unable to fit splines. Details: {e}")
                        return 0.0

                    time_in_region_spline = 0

                    for i in range(1, len(t_values)):
                        previous_t = t_values[i - 1]
                        current_t = t_values[i]
                        time_interval = current_t - previous_t

                        try:
                            previous_x = splev(previous_t, spline_x)
                            previous_y = splev(previous_t, spline_y)
                            current_x = splev(current_t, spline_x)
                            current_y = splev(current_t, spline_y)
                        except Exception as e:
                            st.error(
                                f"Error in spline evaluation at time step {i}. Details: {e}"
                            )
                            continue

                        prev_inside = (x_min <= previous_x <= x_max) and (
                            y_min <= previous_y <= y_max
                        )
                        curr_inside = (x_min <= current_x <= x_max) and (
                            y_min <= current_y <= y_max
                        )

                        if prev_inside and curr_inside:
                            time_in_region_spline += time_interval
                        elif prev_inside or curr_inside:
                            fine_t = np.linspace(previous_t, current_t, num=10)
                            for j in range(1, len(fine_t)):
                                fine_previous_x = splev(fine_t[j - 1], spline_x)
                                fine_previous_y = splev(fine_t[j - 1], spline_y)
                                fine_current_x = splev(fine_t[j], spline_x)
                                fine_current_y = splev(fine_t[j], spline_y)

                                fine_prev_inside = (
                                    x_min <= fine_previous_x <= x_max
                                ) and (y_min <= fine_previous_y <= y_max)
                                fine_curr_inside = (
                                    x_min <= fine_current_x <= x_max
                                ) and (y_min <= fine_current_y <= y_max)

                                if fine_prev_inside and fine_curr_inside:
                                    time_in_region_spline += fine_t[j] - fine_t[j - 1]

                    return time_in_region_spline / 1000.0  # Convert to seconds
            
            # Validate the region before calculation
            if x_min <= x_max and y_min <= y_max:
                # Step 5: Calculate and Display Time in Region
                time_spent = calculate_time_in_region(df, x_min, x_max, y_min, y_max)
                st.write(f"### Time in region: {time_spent:.3f} seconds")

# The "About" Tab
with tabs[1]:
    st.header("How the App Works")
    st.write(
        """
       This app allows you to upload tracking data of an experiment involving animal movement, specifically for mice.
       
       **Here’s how it works**:

       1. **Uploading the File**:
         You start by uploading a `.act` file. The `.act` file should contain tracking data that includes time, x, y, and z coordinates.

       2. **Visualization**:
         Once the file is uploaded, the track will be displayed in a 2D plot. The app will show the trajectory of the mouse, the start point (green), and the end point (orange).
         
         You can customize the region for analysis by specifying values for X and Y boundaries on the sidebar.

       3. **Calculating Time in Region**:
         The region of interest (ROI) is defined by the X and Y boundaries specified on the sidebar.
         
         The app applies **spline interpolation** to accurately model the mouse's trajectory between recorded points and calculates how much time the mouse spent inside the specified ROI.
       
       ### Technical Details:
       - The uploaded data is first processed by extracting only the relevant [Data] section and converting it to a pandas DataFrame.
       - The app uses **splines** calculated by the `scipy.interpolate.splrep` function to create smooth curves approximating the mouse's movement.
       - For each interval, the app checks whether the mouse was inside the region before and after the interval. If the mouse crosses the boundary, it applies a finer time-sampling to estimate the time more accurately.
       """
    )

    st.write(
        "You can view the source code and contribute on our [GitHub Repository](https://github.com/cagriozkurt/tracking_time_in_region)."
    )
