import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Generates a box plot to visualize the age distribution across customer personas.

    Business Value: Helps marketing teams identify if a specific cluster represents
    a younger demographic (e.g., Gen Z) versus an older demographic to tailor campaign messaging.

    Args:
        df (pd.DataFrame): The dataset containing 'Cluster' and 'Age' columns.

    Returns:
        go.Figure: The rendered Plotly box plot figure.
    """
    fig = px.box(
        df,
        x="Cluster",
        y="Age",
        color="Cluster",
        title="Age Demographics by Persona Segment",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(xaxis_title=None)
    return fig


def plot_gender_category_sunburst(df: pd.DataFrame) -> go.Figure:
    """
    Generates a hierarchical sunburst chart mapping Gender -> Category -> Item Purchased.

    Business Value: Allows stakeholders to instantly see which specific items are
    driving the highest revenue for male versus female demographics. Prunes the long
    tail of products to maintain visual clarity and rendering performance.

    Args:
        df (pd.DataFrame): The dataset containing 'Gender', 'Category',
                           'Item Purchased', and 'Purchase Amount (USD)'.

    Returns:
        go.Figure: The rendered Plotly sunburst figure.
    """
    grouped = (
        df.groupby(["Gender", "Category", "Item Purchased"])["Purchase Amount (USD)"]
        .sum()
        .reset_index()
    )

    # Prune: Keep ONLY the top 5 highest-grossing items per category/gender
    grouped = grouped.sort_values(
        ["Gender", "Category", "Purchase Amount (USD)"], ascending=[True, True, False]
    )
    top_grouped = grouped.groupby(["Gender", "Category"]).head(5)

    fig = px.sunburst(
        top_grouped,
        path=["Gender", "Category", "Item Purchased"],
        values="Purchase Amount (USD)",
        color="Gender",
        title="Revenue Drivers by Gender & Category (Top 5 Items)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
    return fig


def plot_subscription_status(df: pd.DataFrame) -> go.Figure:
    """
    Generates a grouped bar chart showing the subscription rate within each cluster.

    Business Value: Essential for identifying which segments hold the most
    loyal, recurring-revenue customers versus one-time buyers.

    Args:
        df (pd.DataFrame): The dataset containing 'Cluster' and 'Subscription Status'.

    Returns:
        go.Figure: The rendered Plotly bar chart figure.
    """
    sub_counts = (
        df.groupby(["Cluster", "Subscription Status"]).size().reset_index(name="Count")
    )

    fig = px.bar(
        sub_counts,
        x="Cluster",
        y="Count",
        color="Subscription Status",
        barmode="group",
        title="Subscription Rate by Persona Segment",
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    fig.update_layout(xaxis_title=None)
    return fig


def plot_elbow_curve(k_values: list, wcss_values: list) -> go.Figure:
    """
    Renders the Elbow Method diagnostic plot for K-Means optimization.

    Mathematical Context: Maps the integer values of 'k' against their computed
    Within-Cluster Sum of Squares (WCSS) to visually identify the inflection point
    (elbow) where marginal variance reduction diminishes.

    Args:
        k_values (list): Integer values representing the number of clusters tested.
        wcss_values (list): The calculated inertia (SSE) for each corresponding k.

    Returns:
        go.Figure: The rendered 2D Plotly line plot.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=wcss_values,
            mode="lines+markers",
            marker=dict(size=8, color="#3b82f6"),
            line=dict(width=2, color="#2563eb"),
        )
    )

    fig.update_layout(
        title="Elbow Method for Determining Optimal k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Sum of Squared Errors (SSE)",
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        template="plotly_white",
    )
    return fig


def plot_cluster_sunburst(df: pd.DataFrame, selected_clusters: list) -> go.Figure:
    """
    Generates a sunburst chart mapping Category and Items within chosen clusters.

    Business Value: Acts as a deep-dive tool after the model is built, allowing
    the user to see exactly what products a specific persona is buying.

    Args:
        df (pd.DataFrame): The dataset containing 'Cluster', 'Category',
                           'Item Purchased', and 'Purchase Amount (USD)'.
        selected_clusters (list): A list of cluster string labels chosen by the user.

    Returns:
        go.Figure: The rendered Plotly sunburst figure.
    """
    filtered_df = df[df["Cluster"].isin(selected_clusters)].copy()

    grouped = (
        filtered_df.groupby(["Cluster", "Category", "Item Purchased"])[
            "Purchase Amount (USD)"
        ]
        .sum()
        .reset_index()
    )

    grouped = grouped.sort_values(
        ["Cluster", "Purchase Amount (USD)"], ascending=[True, False]
    )
    top_grouped = grouped.groupby(["Cluster", "Category"]).head(10)

    fig = px.sunburst(
        top_grouped,
        path=["Cluster", "Category", "Item Purchased"],
        values="Purchase Amount (USD)",
        title="Item Deep-Dive by Selected Clusters",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
    return fig


def plot_seasonal_trends(df: pd.DataFrame) -> go.Figure:
    """
    Generates a grouped bar chart showing revenue generated across different seasons.

    Business Value: Identifies the temporal spending habits of personas (e.g., distinguishing
    between winter holiday shoppers and summer clearance buyers) to optimize campaign timing.

    Args:
        df (pd.DataFrame): The dataset containing 'Cluster', 'Season', and 'Purchase Amount (USD)'.

    Returns:
        go.Figure: The rendered Plotly bar chart figure.
    """
    seasonal_data = (
        df.groupby(["Cluster", "Season"])["Purchase Amount (USD)"].sum().reset_index()
    )

    fig = px.bar(
        seasonal_data,
        x="Season",
        y="Purchase Amount (USD)",
        color="Cluster",
        barmode="group",
        title="Seasonal Revenue by Persona",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(xaxis_title=None, yaxis_title="Total Revenue ($)")
    return fig


def plot_purchase_frequency(df: pd.DataFrame) -> go.Figure:
    """
    Generates a grouped bar chart visualizing the shopping cadence of each persona.

    Business Value: Maps whether personas are habitual buyers (Weekly/Bi-Weekly)
    or one-off shoppers (Annually), informing inventory forecasting and CRM strategies.

    Args:
        df (pd.DataFrame): The dataset containing 'Cluster' and 'Frequency of Purchases'.

    Returns:
        go.Figure: The rendered Plotly bar chart figure.
    """
    freq_counts = (
        df.groupby(["Cluster", "Frequency of Purchases"])
        .size()
        .reset_index(name="Count")
    )

    # Order the x-axis logically from most frequent to least frequent
    frequency_order = [
        "Weekly",
        "Bi-Weekly",
        "Fortnightly",
        "Monthly",
        "Quarterly",
        "Every 3 Months",
        "Annually",
    ]

    fig = px.bar(
        freq_counts,
        x="Frequency of Purchases",
        y="Count",
        color="Cluster",
        barmode="group",
        title="Shopping Frequency & Loyalty by Persona",
        category_orders={"Frequency of Purchases": frequency_order},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(xaxis_title=None, yaxis_title="Number of Customers")
    return fig
