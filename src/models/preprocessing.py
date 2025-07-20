#src/models/preprocessing.py

def create_train_test_split(df, test_size=0.2, random_state=42):
    """
    Create train/test split by taking ~20% of unique boulder names from each grade
    ensuring at least one boulder per grade in test set.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the boulder data
    test_size : float, optional
        Target proportion for test set (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    import numpy as np
    from collections import defaultdict
    from ..features.grade_conversion import difficulty_to_vgrade
    
    np.random.seed(random_state)
    
    # Add V-grade column
    df = df.copy()
    df['v_grade'] = df['grade'].apply(difficulty_to_vgrade)
    
    # Get unique boulder names for each grade
    grade_to_names = defaultdict(set)
    for _, row in df.iterrows():
        grade_to_names[row['v_grade']].add(row['name'])
    
    print("Grade distribution (unique boulder names):")
    for grade in sorted(grade_to_names.keys()):
        print(f"  {grade}: {len(grade_to_names[grade])} unique boulders")
    
    # Select test names for each grade
    test_names = set()
    
    for grade, names in grade_to_names.items():
        names_list = list(names)
        
        # Calculate how many to take for test (at least 1, target ~20%)
        target_count = max(1, int(len(names_list) * test_size))
        
        # Randomly select boulder names for test set
        if len(names_list) >= target_count:
            selected = np.random.choice(names_list, size=target_count, replace=False)
            test_names.update(selected)
            print(f"  {grade}: Selected {len(selected)}/{len(names_list)} boulders for test ({len(selected)/len(names_list)*100:.1f}%)")
        else:
            # If very few boulders, take all
            test_names.update(names_list)
            print(f"  {grade}: Selected ALL {len(names_list)} boulders for test (too few to split)")
    
    print(f"\nTotal test boulder names: {len(test_names)}")
    
    # Create train/test masks
    test_mask = df['name'].isin(test_names)
    train_mask = ~test_mask
    
    # Split the data
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    # Print final statistics
    print(f"\nFinal split:")
    print(f"Training set: {len(train_df)} entries ({train_df['name'].nunique()} unique boulders)")
    print(f"Test set: {len(test_df)} entries ({test_df['name'].nunique()} unique boulders)")
    print(f"Actual test size: {len(test_df)/len(df)*100:.1f}%")
    
    # Show grade distribution in each set
    train_grade_dist = train_df['v_grade'].value_counts().sort_index()
    test_grade_dist = test_df['v_grade'].value_counts().sort_index()
    
    print(f"\nGrade distribution comparison:")
    print(f"{'Grade':<6} {'Train':<8} {'Test':<8} {'Test %':<8}")
    print("-" * 32)
    
    all_grades = sorted(set(train_grade_dist.index) | set(test_grade_dist.index))
    for grade in all_grades:
        train_count = train_grade_dist.get(grade, 0)
        test_count = test_grade_dist.get(grade, 0)
        total_count = train_count + test_count
        test_pct = test_count / total_count * 100 if total_count > 0 else 0
        print(f"{grade:<6} {train_count:<8} {test_count:<8} {test_pct:<8.1f}%")
    
    # Check that all grades are in both sets
    train_grades = set(train_grade_dist.index)
    test_grades = set(test_grade_dist.index)
    
    missing_from_test = train_grades - test_grades
    missing_from_train = test_grades - train_grades
    
    if missing_from_test:
        print(f"\n❌ WARNING: Grades missing from test set: {missing_from_test}")
    if missing_from_train:
        print(f"❌ WARNING: Grades missing from training set: {missing_from_train}")
    
    if not missing_from_test and not missing_from_train:
        print(f"\n✅ SUCCESS: All grades present in both sets!")
    
    # Return in standard format
    X_train = train_df.drop(['grade', 'v_grade'], axis=1)
    y_train = train_df['grade']
    X_test = test_df.drop(['grade', 'v_grade'], axis=1)  
    y_test = test_df['grade']
    
    return X_train, X_test, y_train, y_test