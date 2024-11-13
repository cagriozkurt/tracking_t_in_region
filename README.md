## Mouse Track Visualization and Time Calculation

A **Streamlit-based web application** for **visualizing mouse movement** tracks from an experiment, letting users upload `.act` files, verify mouse's paths spatially, specify regions of interest, and calculate the time spent in those regions.

### Application Overview

The app processes experimental data from **mouse tracking systems**, allowing research teams to visually validate mouse movement paths and compute the time spent in a defined region. The `.act` files not only store mouse coordinates (`x`, `y`, `z`) and time, but include metadata about the experiment. This app focuses on:

1. **Reading `.act` files** to extract the time-series coordinates.
2. Using **spline interpolation** to **smooth the trajectory** of the mouse for accurate time calculations.
3. **Visualizing the track** with custom boundaries for time calculation.
4. Computing the **time spent** in the **region of interest** using high-precision interpolation.

### Usage

Once the app is running in your browser, follow these steps:

1. **Upload Your `.act` File**:
   - Click the **"Upload your .act file"** button on the sidebar and choose your `.act` file containing time, `x`, `y`, and `z` coordinates.

2. **Visualize the Track**:
   - The app automatically parses and visualizes the mouse's trajectory as a dashed line on a 2D plot.
   - The **Start Point** is shown in green, and the **End Point** is in orange.
   - You can adjust the `x_min`, `x_max`, `y_min`, `y_max` in the sidebar to specify the region of interest.

3. **Specify Region of Interest**:
   - Modify the numeric inputs in the sidebar to change the region boundaries.

4. **Time Calculation**:
   - Based on the region specified, the app uses **spline interpolation** to calculate the exact time the mouse spends within the defined region and displays the result.

5. **About Page**:
   - Switch to the **About** tab for a detailed explanation of how the app operates, including technical details about file parsing, spline interpolation, and time interval estimation.
