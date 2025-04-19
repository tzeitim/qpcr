import polars as pl
from scipy import stats


def load_results_DA2(xlsx_ifn):
    import warnings
    standards = {
        "std 0": 0.0,
        "std 1": 100.0,
        "std 2": 10.0,
        "std 3": 1.0,
        "std 4": 0.1,
        "std 5": 0.01,
        "std 6": 0.001,
    }
    dilutions = {
        "6":1e-4,
        "7":1e-4,
        "10":1e-5,
        "11":1e-5,
    }
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")  # Suppress all warnings
        df = (
        pl.read_excel(xlsx_ifn,
                      read_options={"header_row": 24}, )    
        .select(['Well', 'Well Position', 'Sample', 'Target', 'Cq'])
        .with_columns(
            pl.when(pl.col('Cq').cast(pl.Utf8)=="UNDETERMINED")
            .then(pl.lit(None))
            .otherwise(pl.col("Cq"))
            .alias('Cq'))
        .with_columns(pl.col('Cq').cast(pl.Float32))
        .with_columns(pl.col('Well Position').str.extract(r"^.+?(\d+?)$",1).alias("column"))
        .with_columns(dilution = pl.col("column").replace_strict(dilutions, default=None))
                              #.with_columns(dilution = 10**(pl.col("Sample").str.extract(".+1e(.+?)$",1).cast(pl.Float64)))
        .with_columns(dilution = pl.when(pl.col('dilution').is_null()).then(pl.col('Sample').replace(standards)).otherwise(pl.col('dilution')).cast(pl.Float64))
        .group_by("Sample", 'dilution').agg(pl.col('Cq').mean())
        )
    return df#.join(pl.read_excel(xlsx_ann).select("INDEX", "Sample", "Size Tape station"), left_on="Sample", right_on="INDEX", how='full')


def ann_qpcr(df1, xlsx_ann):
   return (df1
           .join(pl.read_excel(xlsx_ann).select("INDEX", "Sample", "Size Tape station"), 
                 left_on="Sample", 
                 right_on="INDEX", 
                 how='full') 
          )


def analyze_qpcr_data(df, workbook=None):
 
    # Identify standard samples
    std_samples = [s for s in df["Sample"].unique() if isinstance(s, str) and s.startswith("std ")]
    
    if not std_samples:
        raise ValueError("No standard samples found in the data")
    
    # Extract standard data for linear regression
    std_data = df.filter(pl.col("Sample").is_in(std_samples))
    
    # Filter out standards with 0 concentration or None Cq values
    std_data = std_data.filter(
        (pl.col("dilution") > 0) & 
        (~pl.col("Cq").is_null())
    )
    
    # Calculate log10 of concentrations for standards
    std_data = std_data.with_columns(log_conc = pl.col("dilution").log10())
    
    # Perform linear regression (Cq vs Log(conc))
    x = std_data["log_conc"].to_numpy()
    y = std_data["Cq"].to_numpy()
    
    # Check if we have enough data points
    if len(x) < 2:
        raise ValueError("Not enough valid standard points for regression")
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value**2
    
    # Calculate qPCR efficiency
    efficiency = 10**(-1/slope) - 1
    
    # Calculate concentrations for all samples based on Cq values
    df = df.with_columns([
        # Only calculate concentration where Cq is not null
        pl.when(~pl.col("Cq").is_null())
          .then(10**((pl.col("Cq") - intercept) / slope) * 1e-3) # to have nM
          .otherwise(None)
          .alias("concentration")
    ])
    
    # Calculate undiluted concentration by dividing by the dilution factor
    df = df.with_columns(
        pl.when(~pl.col("concentration").is_null())
          .then(pl.col("concentration") / pl.col("dilution"))
          .otherwise(None)
          .alias("undiluted_concentration")
    )
    
    # Add size-adjusted concentration (using 399 as reference size)
    df = df.with_columns(
        pl.when(~pl.col("undiluted_concentration").is_null())
          .then(pl.col("undiluted_concentration") * (399 / pl.col("Size Tape station")))
          .otherwise(None)
          .alias("size_adjusted_conc")
    )
    
    # Create a summary dataframe with means of the undiluted concentrations
    summary_df = (
        df.filter(~pl.col("Sample").str.contains("std"))  # Exclude standards
        .group_by(["Sample", "Sample_right", "Size Tape station"])
        .agg([
            pl.col("Cq").mean().alias("mean_Cq"),
            pl.col("undiluted_concentration").mean().alias("mean_undiluted_conc"),
            pl.col("size_adjusted_conc").mean().alias("mean_size_adjusted_conc"),
            pl.col("Cq").count().alias("num_replicates")
        ])
    )
    
    # Return regression metrics and calculated concentrations
    metrics = {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "efficiency": efficiency
    }
    
    # Print metrics
    print(f"Standard Curve Metrics:")
    print(f"  Slope: {metrics['slope']:.4f}")
    print(f"  Y-intercept: {metrics['intercept']:.4f}")
    print(f"  R-squared: {metrics['r_squared']:.4f}")
    print(f"  PCR Efficiency: {metrics['efficiency']*100:.2f}%")
    
    # Write results to Excel if workbook is provided
    if workbook is not None:
        df.sort("Sample", "dilution").write_excel(workbook=workbook)
        summary_df.sort("Sample").write_excel(workbook=workbook.replace(".xlsx", "_summ.xlsx"))
    
    return summary_df, metrics

def main():
    print("""Example:
    xlsx_ifn # DA2 exported Results file
    analyze_qpcr_data(
          ann_qpcr(
            load_results_DA2(xlsx_ifn), 
            'calc-table_20240507_LA_vs_PCRo.xltx'),
        'output.xlsx')
    """)


if __name__ == "__main__":
    main()
