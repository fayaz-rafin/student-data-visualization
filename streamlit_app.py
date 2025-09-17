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
    st.header("Display Options")
    selected_days = st.multiselect("Select days", list(WEEKDAY_TO_FILE.keys()), default=list(WEEKDAY_TO_FILE.keys()))
    show_debug = st.toggle("Show debug info", value=False)


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
            long_df = long_df.sort_values('End Hour Rounded')
            long_df['time_sort'] = long_df['End Hour Rounded']
            # Build an explicit ordered domain of labels from the sorted unique rounded times
            ordered_labels = [format_end_time_label(v) for v in sorted(long_df['End Hour Rounded'].unique())]
            c = alt.Chart(long_df).mark_bar().encode(
                x=alt.X('End Time:N', sort=ordered_labels, title='Class End Time'),
                y=alt.Y('Students:Q', stack='zero', title='Students Available'),
                color=alt.Color('Course:N', legend=alt.Legend(title='Course')),
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


st.divider()
st.markdown(
    "Note: Expected CSV columns include `Hour` (e.g., 13:30), `Dur.` (minutes), `Enr Count`, and `Subj`."
)


