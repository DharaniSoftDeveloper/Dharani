import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('employee_data.csv', parse_dates=['date'])

# Basic preprocessing
df['month'] = df['date'].dt.to_period('M')
monthly_summary = df.groupby(['employee_id', 'month']).agg({
    'hours_worked': 'sum',
    'tasks_completed': 'sum',
    'performance_score': 'mean'
}).reset_index()

# Analyze productivity: tasks per hour
monthly_summary['tasks_per_hour'] = monthly_summary['tasks_completed'] / monthly_summary['hours_worked']

# Plotting productivity trend for an individual employee
employee_id_to_plot = 101  # Example ID
emp_data = monthly_summary[monthly_summary['employee_id'] == employee_id_to_plot]

plt.figure(figsize=(10, 6))
sns.lineplot(data=emp_data, x='month', y='tasks_per_hour', marker='o')
plt.title(f'Productivity Trend - Employee {employee_id_to_plot}')
plt.ylabel('Tasks per Hour')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Department-level analysis
df['month'] = df['date'].dt.to_period('M')
dept_summary = df.groupby(['department', 'month']).agg({
    'hours_worked': 'mean',
    'tasks_completed': 'mean',
    'performance_score': 'mean'
}).reset_index()

# Heatmap of performance trends by department
pivot_table = dept_summary.pivot('department', 'month', 'performance_score')

plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Average Performance Score by Department Over Time')
plt.xlabel('Month')
plt.ylabel('Department')
plt.tight_layout()
plt.show()
