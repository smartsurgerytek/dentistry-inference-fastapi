import optuna
import numpy as np
from test_detntalMeasure_performance import test_detntalMeasure_performance
import yaml
from optuna.visualization import plot_param_importances
# Define the objective function
def objective(trial):
    # Define the hyperparameters search space
    config = {
        'DENTAL_MODEL_THRESHOLD': trial.suggest_float('DENTAL_MODEL_THRESHOLD', 0.3, 0.7),
        'DENTAL_CONTOUR_MODEL_THRESHOLD': trial.suggest_float('DENTAL_CONTOUR_MODEL_THRESHOLD', 0.3, 0.7),
        
        # Mask cleaning parameters
        'DENTAL_CROWN_KERNAL_X': trial.suggest_int('DENTAL_CROWN_KERNAL_X', 1, 5),
        'DENTAL_CROWN_KERNAL_Y': trial.suggest_int('DENTAL_CROWN_KERNAL_Y', 1, 5),
        'DENTAL_CROWN_ITERATION': trial.suggest_int('DENTAL_CROWN_ITERATION', 1, 10),

        'DENTI_KERNAL_X': trial.suggest_int('DENTI_KERNAL_X', 10, 20),
        'DENTI_KERNAL_Y': trial.suggest_int('DENTI_KERNAL_Y', 1, 10),
        'DENTI_ITERATION': trial.suggest_int('DENTI_ITERATION', 1, 10),

        'GUM_KERNAL_X': trial.suggest_int('GUM_KERNAL_X', 1, 5),
        'GUM_KERNAL_Y': trial.suggest_int('GUM_KERNAL_Y', 1, 5),
        'GUM_ITERATION': trial.suggest_int('GUM_ITERATION', 1, 20),

        'DENTAL_CONTOUR_KERNAL_X': trial.suggest_int('DENTAL_CONTOUR_KERNAL_X', 1, 5),
        'DENTAL_CONTOUR_KERNAL_Y': trial.suggest_int('DENTAL_CONTOUR_KERNAL_Y', 1, 5),
        'DENTAL_CONTOUR_ITERATION': trial.suggest_int('DENTAL_CONTOUR_ITERATION', 1, 10),

        'ENAMEL_SKIP_PIXEL_RATIO':trial.suggest_float('ENAMEL_SKIP_PIXEL_RATIO', 0.03, 0.05),
        'ENAMEL_INTERSECTION_THRESHOLD_RATIO':trial.suggest_float('ENAMEL_INTERSECTION_THRESHOLD_RATIO', 0.0005, 0.0015),
        'ENAMEL_NONE_CROWN_REPLACED_HEIGHT_RATIO': trial.suggest_float('ENAMEL_NONE_CROWN_REPLACED_HEIGHT_RATIO', 0.05, 0.15),

        'GUM_LOCATE_GUM_MID_X_THRESHOLD_RATIO': trial.suggest_float('GUM_LOCATE_GUM_MID_X_THRESHOLD_RATIO', 0.03, 0.05),

        # Dental measure algorithm thresholds
        'ROTATION_ANGLE_THRESHOLD': trial.suggest_int('ROTATION_ANGLE_THRESHOLD', 10, 60),
        
        'DENTI_TWO_POINT_TEETH_THRESHOLD_RATIO': trial.suggest_float('DENTI_TWO_POINT_TEETH_THRESHOLD_RATIO', 0.1, 0.3),
        'DENTI_RANGE_FOR_TOOTH_TIP_RATIO': trial.suggest_float('DENTI_RANGE_FOR_TOOTH_TIP_RATIO', 0.04, 0.08),
        'DENTI_RANGE_Y_LEFT_RIGHT_DENTIN_RATIO': trial.suggest_float('DENTI_RANGE_Y_LEFT_RIGHT_DENTIN_RATIO', 0.1, 0.3),
        'DENTI_RANGE_X_LEFT_RIGHT_DENTIN_RATIO': trial.suggest_float('DENTI_RANGE_X_LEFT_RIGHT_DENTIN_RATIO', 0.04, 0.08),
        
        # Validation threshold
        'DISTANCE_THRESHOLD': 300
    }
    
    # Call the evaluation function (replace this with your actual function to evaluate the model)
    score = test_detntalMeasure_performance(config)  # Replace this with your evaluation function
    return score  # Return the score that the optimizer will maximize
def save_best_config(study, trial):
    # Check if this trial is the best so far
    if study.best_trial == trial:
        # Save the best hyperparameters to a YAML file
        best_params = study.best_params
        best_params['best_score'] = study.best_value
        with open('./conf/best_dental_measure_parameters.yaml', 'w') as yaml_file:
            yaml.dump(best_params, yaml_file, default_flow_style=False)
        print("Best hyperparameters saved to best_hyperparameters.yaml")
        fig = plot_param_importances(study)
        fig.write_image("./docs/detental_measure_parameters_searching_correlations.png")

if __name__=='__main__':
    # Create a study to maximize the performance
    plotly_config = {"staticPlot": True}
    study = optuna.create_study(direction='maximize')

    # Start the optimization
    study.optimize(objective, n_trials=1000, n_jobs=6) #callbacks=[save_best_config])  # if setting n_job, comment call back

    # Get the best hyperparameters
    best_params = study.best_params
    best_params['best_score'] = study.best_value
    print("Best hyperparameters:", best_params)
    with open('./conf/best_dental_measure_parameters.yaml', 'w') as yaml_file:
        yaml.dump(best_params, yaml_file, default_flow_style=False)
    fig = plot_param_importances(study)
    fig.show(config=plotly_config)
    fig.write_image("./docs/detental_measure_parameters_searching_correlations.png")    

