import plotly.express as px
import streamlit as st


def make_short_model_label(model_name, max_length=24):
    """
    Shortens long model names for Plotly x-axis labels.

    Parameters
    ----------
    model_name : str
        Original model name.

    max_length : int
        Maximum label length.

    Returns
    -------
    str
        Shortened label.
    """
    model_name = str(model_name)

    if len(model_name) <= max_length:
        return model_name

    return model_name[: max_length - 3] + "..."


def plot_accuracy_breakdown(results_df, has_real_price):
    """
    Plots actual vs predicted prices or predicted prices only.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe.

    has_real_price : bool
        Whether actual prices are available.
    """
    plot_df = results_df.copy()
    plot_df["Model Label"] = plot_df["Model Name"].apply(make_short_model_label)

    if has_real_price:
        plot_df["Actual Price"] = plot_df["Price_USD"]
        plot_df["Predicted Price"] = plot_df["Predicted Price (USD)"]

        plot_df = plot_df.melt(
            id_vars=[
                "Model Name",
                "Model Label",
                "Company Name",
                "Actual Price",
                "Predicted Price",
            ],
            value_vars=["Price_USD", "Predicted Price (USD)"],
            var_name="Price Type",
            value_name="Value",
        )

        plot_df["Price Type"] = plot_df["Price Type"].replace(
            {
                "Price_USD": "Actual Market Price",
                "Predicted Price (USD)": "ML Prediction",
            }
        )

        fig_bar = px.bar(
            plot_df,
            x="Model Label",
            y="Value",
            color="Price Type",
            barmode="group",
            text_auto=".0f",
            custom_data=[
                "Company Name",
                "Model Name",
                "Actual Price",
                "Predicted Price",
                "Price Type",
            ],
            color_discrete_sequence=["#93C5FD", "#1E40AF"],
        )

        fig_bar.update_traces(
            hovertemplate=(
                "<b>Brand:</b> %{customdata[0]}<br>"
                "<b>Model name:</b> %{customdata[1]}<br>"
                "<b>Real price:</b> $%{customdata[2]:,.0f}<br>"
                "<b>Predicted price:</b> $%{customdata[3]:,.0f}<br>"
                "<b>Bar type:</b> %{customdata[4]}"
                "<extra></extra>"
            )
        )

        fig_bar.update_layout(
            legend_title=None,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,
                xanchor="right",
                x=1.0,
            ),
        )

    else:
        fig_bar = px.bar(
            plot_df,
            x="Model Label",
            y="Predicted Price (USD)",
            color="Company Name",
            text_auto=".0f",
            hover_data={
                "Model Name": True,
                "Company Name": True,
                "Model Label": False,
            },
            color_discrete_sequence=px.colors.qualitative.Blues,
        )

        fig_bar.update_layout(
            legend_title="Brand",
        )

    fig_bar.update_layout(
        height=560,
        autosize=True,
        xaxis_title="Phone Model",
        yaxis_title="Price (USD)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(
            l=70,
            r=40,
            t=90,
            b=150,
        ),
        bargap=0.22,
        bargroupgap=0.08,
    )

    fig_bar.update_xaxes(
        tickangle=-30,
        automargin=True,
    )

    fig_bar.update_yaxes(
        automargin=True,
    )

    fig_bar.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    st.plotly_chart(
        fig_bar,
        width="stretch",
        config={"responsive": True},
    )
