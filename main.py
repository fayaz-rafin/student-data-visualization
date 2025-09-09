import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Load all weekday CSV files
csv_files = {
    'Monday': "20250818 Calendar Snapshot(MON FY).csv",
    'Tuesday': "20250818 Calendar Snapshot(TUES FY).csv", 
    'Wednesday': "20250818 Calendar Snapshot(WED FY).csv",
    'Thursday': "20250818 Calendar Snapshot(THURS FY).csv",
    'Friday': "20250818 Calendar Snapshot(FRI FY).csv"
}

# Create a figure with subplots for each day
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

# Process each day
for day_idx, (day_name, csv_file) in enumerate(csv_files.items()):
    if day_idx >= 5:  # Only 5 weekdays
        break
        
    ax = axes[day_idx]
    
    # Load the CSV file
    curves_csv = Path(csv_file)
    curves_df = pd.read_csv(curves_csv)
    
    # Process data for event planning analysis
    curves_df['Hour'] = pd.to_datetime(curves_df['Hour'], format='%H:%M').dt.hour
    curves_df_sorted = curves_df.sort_values('Hour')
    hourly_enrollment = curves_df_sorted.groupby('Hour')['Enr Count'].mean().reset_index()
    
    # Filter out zero enrollment hours for bars
    hourly_enrollment = hourly_enrollment[hourly_enrollment['Enr Count'] > 0]
    
    # Create simple bar chart showing student availability patterns
    bars = ax.bar(hourly_enrollment['Hour'], hourly_enrollment['Enr Count'], 
                  color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Set chart properties
    ax.set_title(f"{day_name} - Student Availability", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Students Available", fontsize=12)
    
    # Set x-axis ticks for complete time range (8 AM to 8 PM)
    complete_hours = list(range(8, 21))  # 8 AM to 8 PM
    ax.set_xticks(complete_hours)
    time_labels = []
    for hour in complete_hours:
        if hour < 12:
            time_labels.append(f"{hour}:00 AM")
        elif hour == 12:
            time_labels.append("12:00 PM")
        else:
            time_labels.append(f"{hour-12}:00 PM")
    ax.set_xticklabels(time_labels, rotation=45, ha='right')
    
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=11)

# Remove the 6th subplot (we only have 5 weekdays)
axes[5].remove()

# Add overall title
fig.suptitle("Weekly Campus Student Availability Analysis", 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()

out_bar = Path("weekly_enrollment_analysis.png")
plt.savefig(out_bar, dpi=150, bbox_inches='tight')
plt.close()

# Print comprehensive weekly analysis
print("Weekly analysis saved as:", out_bar)
print("\n" + "="*80)
print("WEEKLY CAMPUS STUDENT AVAILABILITY ANALYSIS")
print("="*80)

for day_name, csv_file in csv_files.items():
    print(f"\n{day_name.upper()}:")
    print("-" * 40)
    
    # Load the CSV file
    curves_csv = Path(csv_file)
    curves_df = pd.read_csv(curves_csv)
    
    # Process data
    curves_df['Hour'] = pd.to_datetime(curves_df['Hour'], format='%H:%M').dt.hour
    curves_df_sorted = curves_df.sort_values('Hour')
    hourly_enrollment = curves_df_sorted.groupby('Hour')['Enr Count'].mean().reset_index()
    
    # Filter out zero enrollment hours
    hourly_enrollment = hourly_enrollment[hourly_enrollment['Enr Count'] > 0]
    
    # Calculate thresholds for analysis
    peak_threshold = hourly_enrollment['Enr Count'].quantile(0.75)
    low_threshold = hourly_enrollment['Enr Count'].quantile(0.25)
    
    print(f"  Peak Activity Hours (High student activity):")
    peak_hours = hourly_enrollment[hourly_enrollment['Enr Count'] >= peak_threshold]
    for _, row in peak_hours.iterrows():
        hour = row['Hour']
        if hour < 12:
            time_str = f"{hour}:00 AM"
        elif hour == 12:
            time_str = "12:00 PM"
        else:
            time_str = f"{hour-12}:00 PM"
        print(f"    {time_str} - {row['Enr Count']:.0f} students")
    
    print(f"  Low Activity Hours (Good for events):")
    low_hours = hourly_enrollment[hourly_enrollment['Enr Count'] <= low_threshold]
    for _, row in low_hours.iterrows():
        hour = row['Hour']
        if hour < 12:
            time_str = f"{hour}:00 AM"
        elif hour == 12:
            time_str = "12:00 PM"
        else:
            time_str = f"{hour-12}:00 PM"
        print(f"    {time_str} - {row['Enr Count']:.0f} students")
    
    print(f"  All Active Hours:")
    for _, row in hourly_enrollment.iterrows():
        hour = row['Hour']
        if hour < 12:
            time_str = f"{hour}:00 AM"
        elif hour == 12:
            time_str = "12:00 PM"
        else:
            time_str = f"{hour-12}:00 PM"
        print(f"    {time_str} - {row['Enr Count']:.0f} students")

print(f"\n" + "="*80)
print("ANALYSIS SUMMARY:")
print("  • Higher bars = More students in class (busy periods)")
print("  • Lower bars = Fewer students in class (good for events)")
print("  • Only active class periods are shown")
print("  • Use this data to identify optimal event scheduling times")
print("  • Compare across days to find the best event days")
print("="*80)
