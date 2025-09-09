import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.patches as patches

# Load all weekday CSV files
csv_files = {
    'Monday': "20250818 Calendar Snapshot(MON FY).csv",
    'Tuesday': "20250818 Calendar Snapshot(TUES FY).csv", 
    'Wednesday': "20250818 Calendar Snapshot(WED FY).csv",
    'Thursday': "20250818 Calendar Snapshot(THURS FY).csv",
    'Friday': "20250818 Calendar Snapshot(FRI FY).csv"
}

# Create a figure with subplots using nested GridSpec to center bottom row (no middle gap)
fig = plt.figure(figsize=(26, 16), constrained_layout=True)
outer_gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

# Top row: 3 charts spanning the full width
top_gs = outer_gs[0].subgridspec(1, 3)
ax0 = fig.add_subplot(top_gs[0, 0])
ax1 = fig.add_subplot(top_gs[0, 1])
ax2 = fig.add_subplot(top_gs[0, 2])

# Bottom row: 2 charts spanning the full width (centered, no middle whitespace)
bottom_gs = outer_gs[1].subgridspec(1, 2)
ax3 = fig.add_subplot(bottom_gs[0, 0])
ax4 = fig.add_subplot(bottom_gs[0, 1])
axes = [ax0, ax1, ax2, ax3, ax4]

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
    curves_df['Duration'] = curves_df['Dur.'].astype(float) / 60  # Convert minutes to hours
    curves_df['End Hour'] = curves_df['Hour'] + curves_df['Duration']
    
    # Round to nearest 30-minute interval
    curves_df['End Hour Rounded'] = (curves_df['End Hour'] * 2).round() / 2
    
    # Use the 'Subj' column directly from the CSV
    curves_df['Subject'] = curves_df['Subj']
    
    # Create pivot table for stacked bars
    pivot_data = curves_df.pivot_table(
        index='End Hour Rounded', 
        columns='Subject', 
        values='Enr Count', 
        aggfunc='sum', 
        fill_value=0
    )
    
    # Sort by end time
    pivot_data = pivot_data.sort_index()
    
    # Filter to reasonable hours (8 AM to 10 PM)
    pivot_data = pivot_data[
        (pivot_data.index >= 8) & 
        (pivot_data.index <= 22)
    ]
    
    if not pivot_data.empty:
        # Create stacked bar chart
        subjects = pivot_data.columns
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'orange', 'pink', 'lightsteelblue', 'wheat', 'lightcyan']
        
        # Create stacked bars
        bottom = np.zeros(len(pivot_data))
        for i, subject in enumerate(subjects):
            if subject in pivot_data.columns:
                values = pivot_data[subject].values
                color = colors[i % len(colors)]
                ax.bar(range(len(pivot_data)), values, bottom=bottom, 
                      label=subject, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                bottom += values
        
        # Add value labels on bars with dynamic positioning and y-limits to contain labels
        total_values = pivot_data.sum(axis=1)
        max_value = float(total_values.max()) if len(total_values) else 0.0
        label_offset = max(12.0, max_value * 0.06)
        y_max = max_value + label_offset + 10.0
        ax.set_ylim(0, y_max)
        
        for i, total in enumerate(total_values):
            if total > 0:
                ax.text(i, total + label_offset * 0.5, f'{total:.0f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=9, clip_on=True)
        
        # Place legend outside to the right with two columns
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=7, 
                  ncol=2, frameon=True, fancybox=True, shadow=False, borderaxespad=0.0)
    
    # Set chart properties
    ax.set_title(f"{day_name}", fontsize=14, fontweight='bold', pad=22)
    ax.set_xlabel("Class End Time", fontsize=12)
    ax.set_ylabel("Students Available", fontsize=12)
    
    # Set x-axis ticks for rounded end times
    if not pivot_data.empty:
        ax.set_xticks(range(len(pivot_data)))
        time_labels = []
        for hour in pivot_data.index:
            # Convert to proper time format (rounded to nearest 30 minutes)
            hours = int(hour)
            minutes = int((hour - hours) * 60)
            
            if hours < 12:
                time_str = f"{hours:02d}:{minutes:02d} AM"
            elif hours == 12:
                time_str = f"12:{minutes:02d} PM"
            else:
                time_str = f"{hours-12:02d}:{minutes:02d} PM"
            time_labels.append(time_str)
        ax.set_xticklabels(time_labels, rotation=45, ha='right')
        
        # Set x-axis limits to prevent overlapping
        ax.set_xlim(-0.5, len(pivot_data) - 0.5)
    
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=11)

# Add overall title
fig.suptitle("Weekly Students Available by Subject at Class End Times", 
             fontsize=18, fontweight='bold')

# Save figure
out_bar = Path("total_students_available.png")
plt.savefig(out_bar, dpi=150, bbox_inches='tight')
plt.close()

# -----------------------
# Grouped (side-by-side) bars version per subject
# -----------------------
fig2 = plt.figure(figsize=(26, 16), constrained_layout=True)
outer_gs2 = fig2.add_gridspec(2, 1, height_ratios=[1, 1])
# Top row 3 charts
top_gs2 = outer_gs2[0].subgridspec(1, 3)
gx0 = fig2.add_subplot(top_gs2[0, 0])
gx1 = fig2.add_subplot(top_gs2[0, 1])
gx2 = fig2.add_subplot(top_gs2[0, 2])
# Bottom row centered 2 charts
bottom_gs2 = outer_gs2[1].subgridspec(1, 2)
gx3 = fig2.add_subplot(bottom_gs2[0, 0])
gx4 = fig2.add_subplot(bottom_gs2[0, 1])
gaxes = [gx0, gx1, gx2, gx3, gx4]

for day_idx, (day_name, csv_file) in enumerate(csv_files.items()):
    if day_idx >= 5:
        break
    ax = gaxes[day_idx]

    curves_csv = Path(csv_file)
    df = pd.read_csv(curves_csv)
    df['Hour'] = pd.to_datetime(df['Hour'], format='%H:%M').dt.hour
    df['Duration'] = df['Dur.'].astype(float) / 60
    df['End Hour'] = df['Hour'] + df['Duration']
    df['End Hour Rounded'] = (df['End Hour'] * 2).round() / 2
    df['Subject'] = df['Subj']

    pivot = df.pivot_table(index='End Hour Rounded', columns='Subject', values='Enr Count', aggfunc='sum', fill_value=0)
    pivot = pivot.sort_index()
    pivot = pivot[(pivot.index >= 8) & (pivot.index <= 22)]

    if not pivot.empty:
        x = np.arange(len(pivot))
        subjects = list(pivot.columns)
        num_subjects = max(len(subjects), 1)
        bar_width = min(0.8 / num_subjects, 0.12)

        colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'orange', 'pink', 'lightsteelblue', 'wheat', 'lightcyan', 'tan', 'khaki', 'thistle', 'lightsalmon']

        for i, subject in enumerate(subjects):
            values = pivot[subject].values
            offset = (i - (num_subjects - 1) / 2) * bar_width
            ax.bar(x + offset, values, width=bar_width, label=subject, color=colors[i % len(colors)], alpha=0.85, edgecolor='black', linewidth=0.5)

        # Group totals label above each group for readability
        group_totals = pivot.sum(axis=1).values
        max_value = float(group_totals.max()) if len(group_totals) else 0.0
        label_offset = max(12.0, max_value * 0.06)
        ax.set_ylim(0, max_value + label_offset + 10.0)
        for i, total in enumerate(group_totals):
            if total > 0:
                ax.text(x[i], total + label_offset * 0.4, f'{total:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9, clip_on=True)

        ax.set_xticks(x)
        labels = []
        for hour in pivot.index:
            h = int(hour)
            m = int((hour - h) * 60)
            if h < 12:
                t = f"{h:02d}:{m:02d} AM"
            elif h == 12:
                t = f"12:{m:02d} PM"
            else:
                t = f"{h-12:02d}:{m:02d} PM"
            labels.append(t)
        ax.set_xticklabels(labels, rotation=45, ha='right')

        ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=7, ncol=2, frameon=True, fancybox=True)

    ax.set_title(f"{day_name}", fontsize=14, fontweight='bold', pad=22)
    ax.set_xlabel("Class End Time", fontsize=12)
    ax.set_ylabel("Students Available", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=11)

fig2.suptitle("Weekly Students Available by Subject (Grouped Bars)", fontsize=18, fontweight='bold')

out_bar_grouped = Path("total_students_available_grouped.png")
plt.savefig(out_bar_grouped, dpi=150, bbox_inches='tight')
plt.close()

# Print comprehensive weekly analysis
print("Grouped-bars version saved as:", out_bar_grouped)

print("\n" + "="*80)
print("WEEKLY STUDENTS AVAILABLE BY SUBJECT AT CLASS END TIMES")
print("="*80)

for day_name, csv_file in csv_files.items():
    print(f"\n{day_name.upper()}:")
    print("-" * 40)
    
    # Load the CSV file
    curves_csv = Path(csv_file)
    curves_df = pd.read_csv(curves_csv)
    
    # Process data
    curves_df['Hour'] = pd.to_datetime(curves_df['Hour'], format='%H:%M').dt.hour
    curves_df['Duration'] = curves_df['Dur.'].astype(float) / 60  # Convert minutes to hours
    curves_df['End Hour'] = curves_df['Hour'] + curves_df['Duration']
    
    # Round to nearest 30-minute interval
    curves_df['End Hour Rounded'] = (curves_df['End Hour'] * 2).round() / 2
    
    # Use the 'Subj' column directly from the CSV
    curves_df['Subject'] = curves_df['Subj']
    
    # Create pivot table for analysis
    pivot_data = curves_df.pivot_table(
        index='End Hour Rounded', 
        columns='Subject', 
        values='Enr Count', 
        aggfunc='sum', 
        fill_value=0
    )
    
    # Sort by end time
    pivot_data = pivot_data.sort_index()
    
    # Filter to reasonable hours
    pivot_data = pivot_data[
        (pivot_data.index >= 8) & 
        (pivot_data.index <= 22)
    ]
    
    if not pivot_data.empty:
        print(f"  Class End Times and Students Available by Subject:")
        for end_time in pivot_data.index:
            hours = int(end_time)
            minutes = int((end_time - hours) * 60)
            
            if hours < 12:
                time_str = f"{hours:02d}:{minutes:02d} AM"
            elif hours == 12:
                time_str = f"12:{minutes:02d} PM"
            else:
                time_str = f"{hours-12:02d}:{minutes:02d} PM"
            
            print(f"    {time_str}:")
            total_students = 0
            for subject in pivot_data.columns:
                students = pivot_data.loc[end_time, subject]
                if students > 0:
                    print(f"      {subject}: {students:.0f} students")
                    total_students += students
            print(f"      Total: {total_students:.0f} students")
            print()
        
        # Calculate total students
        grand_total = pivot_data.sum().sum()
        print(f"  Grand Total: {grand_total:.0f} students")
        
        # Show subject breakdown
        print(f"\n  Subject Breakdown:")
        subject_totals = pivot_data.sum()
        for subject in subject_totals.index:
            if subject_totals[subject] > 0:
                print(f"    {subject}: {subject_totals[subject]:.0f} students")
    else:
        print("  No classes ending during regular hours on this day")

print(f"\n" + "="*80)
print("ANALYSIS SUMMARY:")
print("  • Each bar shows the breakdown of students by subject")
print("  • Different colors represent different subjects (ADMS, EECS, DIGT, etc.)")
print("  • Stacked bars show total students available at each end time")
print("  • X-axis shows class end times rounded to nearest 30-minute interval")
print("  • Perfect for identifying which subjects have students available when")
print("  • Compare across days to find the best event days")
print("="*80)
