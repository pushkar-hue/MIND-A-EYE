
def get_latest_diagnosis_result(latest_result):
    if not latest_result:
        return {}
        
    result_copy = latest_result.copy()
    
    if result_copy.get('result') in ['Mild', 'Moderate', 'Severe', 'Proliferate_DR']:
        result_copy['condition_type'] = 'diabetic_retinopathy'
        result_copy['recommended_specialist'] = 'Ophthalmologist'
    elif result_copy.get('result') in ['glioma', 'meningioma', 'pituitary']:
        result_copy['condition_type'] = 'brain_tumor'
        result_copy['recommended_specialist'] = 'Neurologist'
    elif result_copy.get('result') == 'No_DR':
        result_copy['condition_type'] = 'normal'
        result_copy['notes'] = 'No signs of diabetic retinopathy detected'
    elif result_copy.get('result') == 'notumor':
        result_copy['condition_type'] = 'normal'
        result_copy['notes'] = 'No brain tumor detected'
        
    return result_copy
