import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import re
from pathlib import Path


# Allow large datasets in Altair (fixes charts not rendering online due to row limits)
alt.data_transformers.disable_max_rows()

# -----------------------------
# Utility and analysis functions
# -----------------------------

WEEKDAY_TO_FILE = {
    'Monday': "20250818 Calendar Snapshot(MON FY).csv",
    'Tuesday': "20250818 Calendar Snapshot(TUES FY).csv",
    'Wednesday': "20250818 Calendar Snapshot(WED FY).csv",
    'Thursday': "20250818 Calendar Snapshot(THURS FY).csv",
    'Friday': "20250818 Calendar Snapshot(FRI FY).csv",
}


def read_csv_safely(file_like_or_path: Path | str) -> pd.DataFrame | None:
    try:
        if isinstance(file_like_or_path, (str, Path)):
            return pd.read_csv(file_like_or_path)
        return pd.read_csv(file_like_or_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return None


def _sanitize_column_name(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


def harmonize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return df with standardized columns and the applied mapping.
    Canonical columns: 'Hour', 'Dur.', 'Enr Count', 'Subj'
    """
    if df is None or df.empty:
        return df, {}
    canonical_targets = {
        'Hour': {
            'hour', 'start', 'starttime', 'time', 'start_time', 'starthour', 'starttime24h', 'start24h'
        },
        'Dur.': {
            'dur', 'duration', 'minutes', 'mins', 'min', 'length', 'durationmin', 'durmin'
        },
        'Enr Count': {
            'enrcount', 'enrollment', 'enrolment', 'enrolled', 'enrollmentcount', 'students', 'count', 'enr'
        },
        'Subj': {
            'subj', 'subject', 'course', 'dept', 'subjectcode', 'major'
        }
    }
    sanitized_to_original = { _sanitize_column_name(c): c for c in df.columns }
    rename_map: dict = {}
    for canon, synonyms in canonical_targets.items():
        for s, original in sanitized_to_original.items():
            if s in synonyms:
                rename_map[original] = canon
                break
    df_renamed = df.rename(columns=rename_map)
    return df_renamed, rename_map


def format_hour_label(hour: int) -> str:
    if hour < 12:
        return f"{hour}:00 AM"
    if hour == 12:
        return "12:00 PM"
    return f"{hour-12}:00 PM"


def format_end_time_label(decimal_hour: float) -> str:
    hours = int(decimal_hour)
    minutes = int(round((decimal_hour - hours) * 60))
    if minutes == 60:
        hours += 1
        minutes = 0
    if hours < 12:
        return f"{hours:02d}:{minutes:02d} AM"
    if hours == 12:
        return f"12:{minutes:02d} PM"
    return f"{hours-12:02d}:{minutes:02d} PM"


def compute_hourly_in_class_means(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or 'Hour' not in df.columns or 'Enr Count' not in df.columns:
        return pd.DataFrame(columns=['Hour', 'Enr Count'])
    df = df.copy()
    df['Hour'] = pd.to_datetime(df['Hour'], format='%H:%M').dt.hour
    hourly = df.sort_values('Hour').groupby('Hour')['Enr Count'].mean().reset_index()
    return hourly[hourly['Enr Count'] > 0]


def compute_endtime_subject_pivot(df: pd.DataFrame) -> pd.DataFrame:
    required = {'Hour', 'Dur.', 'Enr Count', 'Subj'}
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df['Hour'] = pd.to_datetime(df['Hour'], format='%H:%M').dt.hour
    df['Duration'] = df['Dur.'].astype(float) / 60.0
    df['End Hour'] = df['Hour'] + df['Duration']
    df['End Hour Rounded'] = (df['End Hour'] * 2).round() / 2
    df['Subject'] = df['Subj']
    pivot = df.pivot_table(index='End Hour Rounded', columns='Subject', values='Enr Count', aggfunc='sum', fill_value=0)
    pivot = pivot.sort_index()
    # Align with total_students_available.py: show from 8 AM to 10 PM
    pivot = pivot[(pivot.index >= 8) & (pivot.index <= 22)]
    return pivot


def plot_hourly_bar(ax, hourly_enrollment: pd.DataFrame, title: str) -> None:
    bars = ax.bar(hourly_enrollment['Hour'], hourly_enrollment['Enr Count'], color='steelblue', alpha=0.85, edgecolor='navy', linewidth=1)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + max(1.0, height * 0.05), f"{height:.0f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Students In Class (mean)", fontsize=12)
    complete_hours = list(range(8, 21))
    ax.set_xticks(complete_hours)
    ax.set_xticklabels([format_hour_label(h) for h in complete_hours], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)


def plot_stacked_subjects(ax, pivot: pd.DataFrame, title: str) -> None:
    if pivot.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data in the selected range', ha='center', va='center', transform=ax.transAxes)
        return
    subjects = pivot.columns
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'orange', 'pink', 'lightsteelblue', 'wheat', 'lightcyan', 'tan', 'khaki', 'thistle', 'lightsalmon']
    bottom = np.zeros(len(pivot))
    for i, subject in enumerate(subjects):
        values = pivot[subject].values
        color = colors[i % len(colors)]
        ax.bar(range(len(pivot)), values, bottom=bottom, label=subject, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        bottom += values
    total_values = pivot.sum(axis=1)
    max_value = float(total_values.max()) if len(total_values) else 0.0
    label_offset = max(12.0, max_value * 0.06)
    ax.set_ylim(0, max_value + label_offset + 10.0)
    for i, total in enumerate(total_values):
        if total > 0:
            ax.text(i, total + label_offset * 0.4, f"{total:.0f}", ha='center', va='bottom', fontweight='bold', fontsize=9, clip_on=True)
    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels([format_end_time_label(h) for h in pivot.index], rotation=45, ha='right')
    ax.set_xlim(-0.5, len(pivot) - 0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Class End Time", fontsize=12)
    ax.set_ylabel("Students Available", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=8, ncol=2, frameon=True, fancybox=True)


# New helpers to show totals and course-level breakdowns based on end times
def compute_endtime_totals(df: pd.DataFrame) -> pd.DataFrame:
    required = {'Hour', 'Dur.', 'Enr Count'}
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=['End Hour Rounded', 'Total Students'])
    d = df.copy()
    d['Hour'] = pd.to_datetime(d['Hour'], format='%H:%M').dt.hour
    d['Duration'] = d['Dur.'].astype(float) / 60.0
    d['End Hour'] = d['Hour'] + d['Duration']
    d['End Hour Rounded'] = (d['End Hour'] * 2).round() / 2
    totals = (
        d.groupby('End Hour Rounded', as_index=False)['Enr Count']
         .sum()
         .rename(columns={'Enr Count': 'Total Students'})
    )
    totals = totals[(totals['End Hour Rounded'] >= 8) & (totals['End Hour Rounded'] <= 22)]
    return totals.sort_values('End Hour Rounded')


def _find_column_by_synonyms(df: pd.DataFrame, candidates: set[str]) -> str | None:
    if df is None or df.empty:
        return None
    sanitized_to_original = { _sanitize_column_name(c): c for c in df.columns }
    for cand in candidates:
        if cand in sanitized_to_original:
            return sanitized_to_original[cand]
    return None


def compute_endtime_course_pivot(df: pd.DataFrame) -> pd.DataFrame:
    required = {'Hour', 'Dur.', 'Enr Count', 'Subj'}
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    d = df.copy()
    d['Hour'] = pd.to_datetime(d['Hour'], format='%H:%M').dt.hour
    d['Duration'] = d['Dur.'].astype(float) / 60.0
    d['End Hour'] = d['Hour'] + d['Duration']
    d['End Hour Rounded'] = (d['End Hour'] * 2).round() / 2
    subj_col = 'Subj'
    crs_num_col = _find_column_by_synonyms(d, {'crsnum', 'coursenumber', 'coursecode', 'courseno'}) or ('Crs Num' if 'Crs Num' in d.columns else None)
    title_col = _find_column_by_synonyms(d, {'coursetitle', 'title'}) or ('Course Title' if 'Course Title' in d.columns else None)
    if crs_num_col is not None and title_col is not None:
        d['Course'] = d[subj_col].astype(str).str.strip() + ' ' + d[crs_num_col].astype(str).str.strip() + ' - ' + d[title_col].astype(str).str.strip()
    elif crs_num_col is not None:
        d['Course'] = d[subj_col].astype(str).str.strip() + ' ' + d[crs_num_col].astype(str).str.strip()
    else:
        d['Course'] = d[subj_col].astype(str).str.strip()
    pivot = d.pivot_table(index='End Hour Rounded', columns='Course', values='Enr Count', aggfunc='sum', fill_value=0)
    pivot = pivot.sort_index()
    pivot = pivot[(pivot.index >= 8) & (pivot.index <= 22)]
    return pivot


# -----------------------------
# Program presence estimation
# -----------------------------
# Load major data from CSV
def load_major_data() -> tuple[dict[str, str], dict[str, str]]:
    """Load major and course mappings from the CSV file."""
    try:
        major_csv_path = Path("Checklists/Markham_Majors___Y1_2_Courses__2025_26_.csv")
        if not major_csv_path.exists():
            # Fallback to original mapping if CSV not found
            return get_fallback_mapping()
        
        df = pd.read_csv(major_csv_path)
        
        # Create subject to program mapping
        program_mapping = {}
        major_categories = {}
        
        for _, row in df.iterrows():
            major = row['Major']
            degree = row['Degree']
            course_code = row['Course Code']
            
            if pd.notna(course_code) and course_code.strip():
                # Extract subject code from course code (e.g., "AP/SPRT 1010" -> "SPRT")
                if '/' in course_code:
                    subject = course_code.split('/')[1].split()[0]
                else:
                    subject = course_code.split()[0]
                
                # Map subject to program
                if subject not in program_mapping:
                    program_mapping[subject] = major
                
                # Create major categories
                if major not in major_categories:
                    major_categories[major] = major
        
        # Add some common subjects that might not be in the CSV
        additional_mappings = {
            'EECS': 'Computer Science for Software Development (CSSD)',
            'DIGT': 'Digital Technologies',
            'ENG': 'First Year Engineering Core',
            'BIOL': 'First Year Science Program',
            'CHEM': 'First Year Science Program',
            'MATH': 'First Year Science Program',
            'PHYS': 'First Year Science Program',
        }
        
        program_mapping.update(additional_mappings)
        
        return program_mapping, major_categories
        
    except Exception as e:
        print(f"Error loading major data: {e}")
        return get_fallback_mapping()

def get_fallback_mapping() -> tuple[dict[str, str], dict[str, str]]:
    """Fallback mapping if CSV loading fails."""
    program_mapping = {
        'CMDS': 'Communication & Media Studies (CMDS)',
        'CSSD': 'Computer Science for Software Development (CSSD)',
        'EECS': 'Computer Science for Software Development (CSSD)',
        'CRTE': 'Creative Technologies (CRTE)',
        'DIGT': 'Digital Technologies',
        'ADMS': 'Entrepreneurship & Innovation (ENTP)',
        'SPRT': 'Sport Management (BSM)',
        'FINT': 'Financial Technologies (FINT)',
        'ENG': 'First Year Engineering Core',
        'BIOL': 'First Year Science Program',
        'CHEM': 'First Year Science Program',
        'MATH': 'First Year Science Program',
        'PHYS': 'First Year Science Program',
    }
    
    major_categories = {
        'Communication & Media Studies (CMDS)': 'Communication & Media Studies (CMDS)',
        'Computer Science for Software Development (CSSD)': 'Computer Science for Software Development (CSSD)',
        'Creative Technologies (CRTE)': 'Creative Technologies (CRTE)',
        'Entrepreneurship & Innovation (ENTP)': 'Entrepreneurship & Innovation (ENTP)',
        'Sport Management (BSM)': 'Sport Management (BSM)',
        'Financial Technologies (FINT)': 'Financial Technologies (FINT)',
        'Digital Technologies': 'Digital Technologies',
        'First Year Engineering Core': 'First Year Engineering Core',
        'First Year Science Program': 'First Year Science Program',
        'Unmapped/Other': 'Other Programs'
    }
    
    return program_mapping, major_categories

# Load the mappings
PROGRAM_BY_SUBJECT, MAJOR_CATEGORIES = load_major_data()

ANCHOR_SUBJECTS = {'CMDS','CSSD','EECS','CRTE','DIGT','ADMS','SPRT','FINT'}


def compute_students_per_major(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total students per major based on course enrollments."""
    if df is None or df.empty or 'Subj' not in df.columns or 'Enr Count' not in df.columns:
        return pd.DataFrame(columns=['Major', 'Total Students', 'Course Count'])
    
    d = df.copy()
    d['Program'] = d['Subj'].map(PROGRAM_BY_SUBJECT).fillna('Unmapped/Other')
    
    # Group by major and sum enrollments
    major_stats = d.groupby('Program', as_index=False).agg({
        'Enr Count': 'sum',
        'Subj': 'nunique'  # Count unique subjects per major
    }).rename(columns={
        'Enr Count': 'Total Students',
        'Subj': 'Course Count'
    })
    
    # Sort by total students descending
    major_stats = major_stats.sort_values('Total Students', ascending=False)
    
    return major_stats


def compute_program_presence(df: pd.DataFrame) -> pd.DataFrame:
    required = {'Hour', 'Dur.', 'Enr Count', 'Subj'}
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=['End Hour Rounded','Program','Students'])
    d = df.copy()
    # If Campus exists, prefer Markham only
    if 'Campus' in d.columns:
        d = d[d['Campus'].astype(str).str.upper().eq('MA')]
    d['Hour'] = pd.to_datetime(d['Hour'], format='%H:%M').dt.hour
    d['Duration'] = d['Dur.'].astype(float) / 60.0
    d['End Hour'] = d['Hour'] + d['Duration']
    d['End Hour Rounded'] = (d['End Hour'] * 2).round() / 2
    d = d[(d['End Hour Rounded'] >= 8) & (d['End Hour Rounded'] <= 22)]
    d['Program'] = d['Subj'].map(PROGRAM_BY_SUBJECT).fillna('Unmapped/Other')
    d['is_anchor'] = d['Subj'].isin(ANCHOR_SUBJECTS)

    anchor = (d[d['is_anchor']]
              .groupby(['End Hour Rounded','Program'], as_index=False)['Enr Count'].sum()
              .rename(columns={'Enr Count':'AnchorTotal'}))
    time_anchor = anchor.groupby('End Hour Rounded', as_index=False)['AnchorTotal'].sum().rename(columns={'AnchorTotal':'AllAnchorAtTime'})
    anchor = anchor.merge(time_anchor, on='End Hour Rounded', how='left')
    anchor['Weight'] = anchor['AnchorTotal'] / anchor['AllAnchorAtTime'].replace({0: pd.NA})
    anchor['Weight'] = anchor['Weight'].fillna(0.0)

    service = d[~d['is_anchor']].copy()
    service_total_by_time = service.groupby('End Hour Rounded', as_index=False)['Enr Count'].sum().rename(columns={'Enr Count':'ServiceAtTime'})

    program_presence = anchor[['End Hour Rounded','Program','AnchorTotal']].rename(columns={'AnchorTotal':'Students'})
    alloc = anchor[['End Hour Rounded','Program','Weight']].merge(service_total_by_time, on='End Hour Rounded', how='left')
    alloc['Allocated'] = alloc['Weight'] * alloc['ServiceAtTime'].fillna(0)
    alloc = alloc[['End Hour Rounded','Program','Allocated']].rename(columns={'Allocated':'Students'})
    presence = pd.concat([program_presence, alloc], ignore_index=True)
    presence = presence.groupby(['End Hour Rounded','Program'], as_index=False)['Students'].sum()

    if 'Unmapped/Other' in d['Program'].unique():
        unmapped = d[d['Program'].eq('Unmapped/Other')].groupby('End Hour Rounded', as_index=False)['Enr Count'].sum()
        if not unmapped.empty:
            tmp = unmapped.copy()
            tmp['Program'] = 'Unmapped/Other'
            tmp = tmp.rename(columns={'Enr Count':'Students'})
            presence = pd.concat([presence, tmp], ignore_index=True)

    return presence.sort_values(['End Hour Rounded','Program'])

# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(page_title="Campus Availability Analytics", layout="wide")

st.title("Campus Class Availability Analytics")
st.caption("Interactive exploration of student availability by hour and by subject at class end times.")

with st.sidebar:
    st.header("Data Sources")
    st.write("Upload CSVs per weekday or use the sample files present in the repo.")
    use_sample = st.toggle("Use sample CSVs from repository", value=True)
    uploaded = {}
    if not use_sample:
        for day in WEEKDAY_TO_FILE.keys():
            uploaded[day] = st.file_uploader(f"Upload {day} CSV", type=["csv"], key=f"uploader_{day}")
    st.divider()
    st.header("Major Filter")
    st.write("Filter data by specific majors based on degree checklists:")
    selected_majors = st.multiselect(
        "Select majors to include", 
        list(MAJOR_CATEGORIES.keys()),
        default=list(MAJOR_CATEGORIES.keys()),
        help="Select specific majors to filter the data. Based on Markham Majors CSV data from the Checklists folder."
    )
    st.divider()
    st.header("Display Options")
    selected_days = st.multiselect("Select days", list(WEEKDAY_TO_FILE.keys()), default=list(WEEKDAY_TO_FILE.keys()))
    max_courses = st.slider("Max courses in course chart", min_value=10, max_value=100, value=30, step=5, help="Top courses by total students; others aggregated to 'Other'")
    legend_limit = st.slider("Legend items (max)", min_value=5, max_value=30, value=12, step=1, help="Limit legend entries to keep it readable")
    show_debug = st.toggle("Show debug info", value=False)

# Display current filter status
if selected_majors:
    st.info(f"📊 **Currently showing data for:** {', '.join([MAJOR_CATEGORIES[major] for major in selected_majors])}")
else:
    st.warning("⚠️ **No majors selected** - showing all data")


def filter_dataframe_by_majors(df: pd.DataFrame, selected_majors: list[str]) -> pd.DataFrame:
    """Filter dataframe to include only courses from selected majors."""
    if df is None or df.empty or not selected_majors:
        return df
    
    # Map subjects to programs
    df['Program'] = df['Subj'].map(PROGRAM_BY_SUBJECT).fillna('Unmapped/Other')
    
    # Filter by selected majors
    filtered_df = df[df['Program'].isin(selected_majors)].copy()
    
    # Remove the temporary Program column
    if 'Program' in filtered_df.columns:
        filtered_df = filtered_df.drop('Program', axis=1)
    
    return filtered_df


def load_day_dataframe(day_name: str) -> pd.DataFrame | None:
    if not selected_days or day_name not in selected_days:
        return None
    if use_sample:
        csv_path = Path(WEEKDAY_TO_FILE[day_name])
        if not csv_path.exists():
            return None
        df = read_csv_safely(csv_path)
        if df is None or df.empty:
            st.info(f"No rows in sample CSV for {day_name}.")
            return None
        df, mapping = harmonize_columns(df)
        if show_debug and mapping:
            st.caption(f"Column mapping: {mapping}")
        required = {'Hour', 'Dur.', 'Enr Count', 'Subj'}
        if not required.issubset(df.columns):
            if show_debug:
                st.warning(f"{day_name}: Missing required columns. Found: {list(df.columns)}")
            return None
        # Apply major filter
        df = filter_dataframe_by_majors(df, selected_majors)
        return df
    file_like = uploaded.get(day_name)
    if file_like is None:
        return None
    df = read_csv_safely(file_like)
    if df is None or df.empty:
        st.info(f"Uploaded CSV for {day_name} is empty.")
        return None
    df, mapping = harmonize_columns(df)
    if show_debug and mapping:
        st.caption(f"Column mapping: {mapping}")
    required = {'Hour', 'Dur.', 'Enr Count', 'Subj'}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        st.warning(f"{day_name}: Missing columns {sorted(list(missing))}. Found: {list(df.columns)}")
        if show_debug:
            st.dataframe(df.head())
        return None
    # Apply major filter
    df = filter_dataframe_by_majors(df, selected_majors)
    return df


# Add student count per major section
st.subheader("📈 Students per Major")
st.write("Total student enrollment by major based on course data:")

# Collect data from all selected days
all_major_data = []
for day_name in selected_days:
    df = load_day_dataframe(day_name)
    if df is not None and not df.empty:
        major_stats = compute_students_per_major(df)
        if not major_stats.empty:
            major_stats['Day'] = day_name
            all_major_data.append(major_stats)

if all_major_data:
    # Combine all days and aggregate
    combined_major_data = pd.concat(all_major_data, ignore_index=True)
    
    # Aggregate across all days
    daily_totals = combined_major_data.groupby('Program', as_index=False).agg({
        'Total Students': 'sum',
        'Course Count': 'mean'  # Average course count across days
    }).round(0)
    
    # Sort by total students
    daily_totals = daily_totals.sort_values('Total Students', ascending=False)
    
    # Create two columns for display
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Total Students by Major (All Days Combined)**")
        if not daily_totals.empty:
            # Create a bar chart
            chart_data = daily_totals.copy()
            chart_data['Major_Short'] = chart_data['Program'].apply(
                lambda x: MAJOR_CATEGORIES.get(x, x)[:30] + "..." if len(MAJOR_CATEGORIES.get(x, x)) > 30 else MAJOR_CATEGORIES.get(x, x)
            )
            
            c = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Total Students:Q', title='Total Students'),
                y=alt.Y('Major_Short:N', sort='-x', title='Major'),
                tooltip=[
                    alt.Tooltip('Program:N', title='Major'),
                    alt.Tooltip('Total Students:Q', title='Total Students', format='.0f'),
                    alt.Tooltip('Course Count:Q', title='Avg Courses', format='.0f')
                ],
                color=alt.value('#4682B4')
            ).properties(
                height=400,
                title="Student Enrollment by Major"
            ).interactive()
            
            st.altair_chart(c, use_container_width=True)
        else:
            st.info("No data available for the selected majors.")
    
    with col2:
        st.write("**📋 Detailed Breakdown**")
        if not daily_totals.empty:
            # Format the data for display
            display_data = daily_totals.copy()
            display_data['Major'] = display_data['Program'].apply(
                lambda x: MAJOR_CATEGORIES.get(x, x)
            )
            display_data = display_data[['Major', 'Total Students', 'Course Count']]
            display_data['Course Count'] = display_data['Course Count'].astype(int)
            
            # Add percentage column
            total_students = display_data['Total Students'].sum()
            display_data['Percentage'] = (display_data['Total Students'] / total_students * 100).round(1)
            
            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Major": st.column_config.TextColumn("Major", width="medium"),
                    "Total Students": st.column_config.NumberColumn("Total Students", format="%d"),
                    "Course Count": st.column_config.NumberColumn("Avg Courses", format="%d"),
                    "Percentage": st.column_config.NumberColumn("Percentage", format="%.1f%%")
                }
            )
            
            # Summary statistics
            st.metric("Total Students Across All Majors", f"{total_students:,.0f}")
            st.metric("Number of Majors", len(display_data))
        else:
            st.info("No data available for the selected majors.")
else:
    st.warning("No data available for the selected days and majors.")


def load_day_dataframe(day_name: str) -> pd.DataFrame | None:
    if not selected_days or day_name not in selected_days:
        return None
    if use_sample:
        csv_path = Path(WEEKDAY_TO_FILE[day_name])
        if not csv_path.exists():
            return None
        df = read_csv_safely(csv_path)
        if df is None or df.empty:
            st.info(f"No rows in sample CSV for {day_name}.")
            return None
        df, mapping = harmonize_columns(df)
        if show_debug and mapping:
            st.caption(f"Column mapping: {mapping}")
        required = {'Hour', 'Dur.', 'Enr Count', 'Subj'}
        if not required.issubset(df.columns):
            if show_debug:
                st.warning(f"{day_name}: Missing required columns. Found: {list(df.columns)}")
            return None
        # Apply major filter
        df = filter_dataframe_by_majors(df, selected_majors)
        return df
    file_like = uploaded.get(day_name)
    if file_like is None:
        return None
    df = read_csv_safely(file_like)
    if df is None or df.empty:
        st.info(f"Uploaded CSV for {day_name} is empty.")
        return None
    df, mapping = harmonize_columns(df)
    if show_debug and mapping:
        st.caption(f"Column mapping: {mapping}")
    required = {'Hour', 'Dur.', 'Enr Count', 'Subj'}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        st.warning(f"{day_name}: Missing columns {sorted(list(missing))}. Found: {list(df.columns)}")
        if show_debug:
            st.dataframe(df.head())
        return None
    # Apply major filter
    df = filter_dataframe_by_majors(df, selected_majors)
    return df


st.subheader("1) Students Available at Class End Times (Total)")
hourly_any = False
plots_per_row = 3
day_chunks = [list(WEEKDAY_TO_FILE.keys())[i:i+plots_per_row] for i in range(0, len(WEEKDAY_TO_FILE), plots_per_row)]
for row_idx, day_group in enumerate(day_chunks):
    cols = st.columns(len(day_group))
    for col_idx, day in enumerate(day_group):
        with cols[col_idx]:
            df = load_day_dataframe(day)
            if df is None:
                st.info(f"No data for {day}.")
                continue
            totals_df = compute_endtime_totals(df)
            if totals_df.empty:
                st.info(f"No class end times in range for {day}.")
                continue
            hourly_any = True
            chart_df = totals_df.copy()
            chart_df['End Time'] = chart_df['End Hour Rounded'].apply(format_end_time_label)
            chart_df['time_sort'] = chart_df['End Hour Rounded']
            c = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('End Time:N', sort=alt.SortField(field='time_sort', order='ascending'), title='Class End Time'),
                y=alt.Y('Total Students:Q', title='Students Available (total)'),
                tooltip=[
                    alt.Tooltip('End Time:N', title='End Time'),
                    alt.Tooltip('Total Students:Q', title='Total Students', format='.0f'),
                ],
                color=alt.value('#4682B4')
            ).properties(title=f"{day}").interactive()
            st.altair_chart(c, use_container_width=True)

if not hourly_any:
    st.warning("Upload/select at least one day with valid data to see hourly charts.")


st.subheader("2) Students Available at Class End Times by Course")
stacked_any = False
for row_idx, day_group in enumerate(day_chunks):
    cols = st.columns(len(day_group))
    for col_idx, day in enumerate(day_group):
        with cols[col_idx]:
            df = load_day_dataframe(day)
            if df is None:
                st.info(f"No data for {day}.")
                continue
            pivot = compute_endtime_course_pivot(df)
            if pivot.empty:
                st.info(f"No class end times in range for {day}.")
                continue
            stacked_any = True
            long_df = pivot.reset_index().melt(id_vars=['End Hour Rounded'], var_name='Course', value_name='Students')
            long_df = long_df[long_df['Students'] > 0]
            long_df['End Time'] = long_df['End Hour Rounded'].apply(format_end_time_label)
            totals = long_df.groupby('End Time', as_index=False)['Students'].sum().rename(columns={'Students':'Total Students'})
            long_df = long_df.merge(totals, on='End Time', how='left')
            # Limit number of courses to top N by total students; aggregate rest into 'Other'
            course_totals = long_df.groupby('Course', as_index=False)['Students'].sum().sort_values('Students', ascending=False)
            keep_courses = set(course_totals['Course'].head(max_courses))
            long_df['Course'] = long_df['Course'].where(long_df['Course'].isin(keep_courses), other='Other')
            long_df = long_df.groupby(['End Hour Rounded', 'End Time', 'Course'], as_index=False).agg({'Students':'sum', 'Total Students':'first'})
            # Build legend values: top legend_limit courses plus 'Other' (if present)
            legend_totals = long_df.groupby('Course', as_index=False)['Students'].sum().sort_values('Students', ascending=False)
            legend_courses = [c for c in legend_totals['Course'].tolist() if c != 'Other'][:legend_limit]
            if 'Other' in legend_totals['Course'].values:
                legend_courses = legend_courses + ['Other']
            long_df = long_df.sort_values('End Hour Rounded')
            long_df['time_sort'] = long_df['End Hour Rounded']
            # Build an explicit ordered domain of labels from the sorted unique rounded times
            ordered_labels = [format_end_time_label(v) for v in sorted(long_df['End Hour Rounded'].unique())]
            c = alt.Chart(long_df).mark_bar().encode(
                x=alt.X('End Time:N', sort=ordered_labels, title='Class End Time'),
                y=alt.Y('Students:Q', stack='zero', title='Students Available'),
                color=alt.Color('Course:N', legend=alt.Legend(title='Course', values=legend_courses, columns=1)),
                tooltip=[
                    alt.Tooltip('End Time:N', title='End Time'),
                    alt.Tooltip('Course:N', title='Course'),
                    alt.Tooltip('Students:Q', title='Students', format='.0f'),
                    alt.Tooltip('Total Students:Q', title='Total at Time', format='.0f'),
                ]
            ).properties(title=f"{day}").interactive()
            st.altair_chart(c, use_container_width=True)

if not stacked_any:
    st.warning("Upload/select at least one day with valid data to see subject breakdown charts.")


st.subheader("3) Students Present by Program (Markham)")
program_any = False
for row_idx, day_group in enumerate(day_chunks):
    cols = st.columns(len(day_group))
    for col_idx, day in enumerate(day_group):
        with cols[col_idx]:
            df = load_day_dataframe(day)
            if df is None:
                st.info(f"No data for {day}.")
                continue
            presence = compute_program_presence(df)
            if presence.empty:
                st.info(f"No program presence derived for {day}.")
                continue
            program_any = True
            presence['End Time'] = presence['End Hour Rounded'].apply(format_end_time_label)
            # Legend limiting: top programs + 'Unmapped/Other'
            prog_totals = presence.groupby('Program', as_index=False)['Students'].sum().sort_values('Students', ascending=False)
            legend_programs = [p for p in prog_totals['Program'].tolist() if p != 'Unmapped/Other'][:legend_limit]
            if 'Unmapped/Other' in prog_totals['Program'].values:
                legend_programs = legend_programs + ['Unmapped/Other']
            presence = presence.sort_values('End Hour Rounded')
            ordered_labels = [format_end_time_label(v) for v in sorted(presence['End Hour Rounded'].unique())]
            c = alt.Chart(presence).mark_bar().encode(
                x=alt.X('End Time:N', sort=ordered_labels, title='Class End Time'),
                y=alt.Y('Students:Q', stack='zero', title='Estimated Students Present'),
                color=alt.Color('Program:N', legend=alt.Legend(title='Program', values=legend_programs, columns=1)),
                tooltip=[
                    alt.Tooltip('End Time:N', title='End Time'),
                    alt.Tooltip('Program:N', title='Program'),
                    alt.Tooltip('Students:Q', title='Students', format='.0f'),
                ]
            ).properties(title=f"{day}").interactive()
            st.altair_chart(c, use_container_width=True)

if not program_any:
    st.info("Upload/select at least one day with valid data to see program presence charts.")


st.divider()
st.markdown(
    "Note: Expected CSV columns include `Hour` (e.g., 13:30), `Dur.` (minutes), `Enr Count`, and `Subj`."
)


