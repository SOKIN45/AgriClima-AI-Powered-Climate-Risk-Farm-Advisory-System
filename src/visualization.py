import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


class ClimateVisualizer:
    def __init__(self, df):
        self.df = df

    def plot_monthly_trends(self):
        """Plot monthly rainfall and temperature trends"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Rainfall plot
        ax1.bar(self.df['Month'], self.df['Rainfall_Amount'], color='skyblue', alpha=0.7)
        ax1.set_title('Monthly Rainfall Patterns in Lumbia-El Salvador')
        ax1.set_ylabel('Rainfall (mm)')
        ax1.grid(True, alpha=0.3)

        # Temperature plot
        ax2.plot(self.df['Month'], self.df['Max_Temperature'], marker='o', label='Max Temp', color='red')
        ax2.plot(self.df['Month'], self.df['Min_Temperature'], marker='o', label='Min Temp', color='blue')
        ax2.plot(self.df['Month'], self.df['Mean_Temperature'], marker='s', label='Mean Temp', color='green')
        ax2.set_title('Monthly Temperature Patterns')
        ax2.set_ylabel('Temperature (°C)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_interactive_plots(self):
        """Create interactive Plotly visualizations"""
        # Rainfall and Temperature correlation
        fig1 = px.scatter(self.df, x='Mean_Temperature', y='Rainfall_Amount',
                          size='Number_of_Rainy_Days', color='Month',
                          title='Temperature vs Rainfall Correlation',
                          hover_data=['Month', 'Relative_Humidity'])

        # Monthly patterns
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=self.df['Month'], y=self.df['Rainfall_Amount'],
                              name='Rainfall', marker_color='lightblue'))
        fig2.add_trace(go.Scatter(x=self.df['Month'], y=self.df['Mean_Temperature'],
                                  name='Mean Temperature', yaxis='y2',
                                  line=dict(color='red')))
        fig2.update_layout(
            title='Monthly Climate Patterns',
            yaxis=dict(title='Rainfall (mm)'),
            yaxis2=dict(title='Temperature (°C)', overlaying='y', side='right'),
            xaxis=dict(title='Month')
        )

        return fig1, fig2

    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)

            plt.figure(figsize=(10, 8))
            plt.barh(feature_imp['feature'], feature_imp['importance'])
            plt.title('Feature Importance')
            plt.tight_layout()
            return plt.gcf()