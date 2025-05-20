# src/visualization/plots.py (continued)

def plot_grade_distribution(data, output_file=None, most_popular_only=True,
                            title=None, figsize=(12, 6)):
    """
    Create a sorted grade distribution plot from Kilterboard data.

    Parameters:
    -----------
    data : pandas.DataFrame or dict
        Either a DataFrame with climb data or the output from create_data_profile()
    output_file : str, optional
        Path to save the plot (if None, plot will be displayed)
    most_popular_only : bool, optional
        Whether to count only the most popular angle for each climb (default: True)
    title : str, optional
        Custom title for the plot (if None, a default title will be used)
    figsize : tuple, optional
        Figure size as (width, height)

    Returns:
    --------
    dict
        Dictionary with grade distribution data
    """
    # Define grade order for sorting
    grade_order = ['VB', 'V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                  'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16+']

    # Check if data is already processed
    if isinstance(data, dict) and ('popular_grades_count' in data or 'all_grades_count' in data):
        # Use pre-calculated data
        if most_popular_only and 'popular_grades_count' in data:
            grades_count = data['popular_grades_count']
        elif 'all_grades_count' in data:
            grades_count = data['all_grades_count']
        else:
            print("Warning: Using available grade count data")
            grades_count = data.get('popular_grades_count', data.get('all_grades_count', {}))
    else:
        # We need to process the DataFrame
        if not isinstance(data, pd.DataFrame):
            print("Error: Input must be either a DataFrame or output from create_data_profile()")
            return None

        # Process data to get grade counts
        from ..data.processing import parse_climb_stats, find_most_popular_setup
        from ..features.grade_conversion import difficulty_to_vgrade

        if most_popular_only:
            # Count each boulder only once at its most popular angle
            grades = []
            for _, row in data.iterrows():
                popular_setup = find_most_popular_setup(row.get('climb_stats', []))
                if popular_setup:
                    difficulty = popular_setup.get('difficulty_average')
                    if difficulty is not None:
                        vgrade = difficulty_to_vgrade(difficulty)
                        grades.append(vgrade)
        else:
            # Count all setups at all angles
            grades = []
            for _, row in data.iterrows():
                stats_list = parse_climb_stats(row.get('climb_stats', []))
                for stats in stats_list:
                    difficulty = stats.get('difficulty_average')
                    if difficulty is not None:
                        vgrade = difficulty_to_vgrade(difficulty)
                        grades.append(vgrade)

        # Create the counter dictionary
        from collections import Counter
        grades_count = Counter(grades)

    # Filter out any 'N/A' grades
    if 'N/A' in grades_count:
        del grades_count['N/A']

    # Create sorted list of grades and counts
    sorted_grades = []
    sorted_counts = []

    # Sort according to grade_order
    for grade in grade_order:
        if grade in grades_count:
            sorted_grades.append(grade)
            sorted_counts.append(grades_count[grade])

    # Create the plot
    plt.figure(figsize=figsize)

    # Create bars with a nice color gradient from light to dark
    bar_colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(sorted_grades)))

    # Plot bars
    bars = plt.bar(sorted_grades, sorted_counts, color=bar_colors)

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}',
                 ha='center', va='bottom', rotation=0, fontsize=9)

    # Set title
    if title is None:
        if most_popular_only:
            title = 'Distribution of Boulder Problems by V-Grade (Most Popular Angle)'
        else:
            title = 'Distribution of Boulder Problems by V-Grade (All Angles)'

    plt.xlabel('V-Grade', fontsize=12)
    plt.ylabel('Number of Problems', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Grade distribution plot saved to {output_file}")
    else:
        plt.show()

    plt.close()

    # Return the data
    return {
        'sorted_grades': sorted_grades,
        'sorted_counts': sorted_counts,
        'grades_count': dict(grades_count)
    }