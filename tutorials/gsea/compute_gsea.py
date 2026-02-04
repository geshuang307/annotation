import os
import pandas as pd
import decoupler
from pathlib import Path
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import issparse
import seaborn as sns

def gmt_to_decoupler(pth: Path) -> pd.DataFrame:
    """
    Parse a gmt file to a decoupler pathway dataframe.
    """
    from itertools import chain, repeat

    pathways = {}

    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    return pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["geneset", "genesymbol"],
    )

def compute_gsea_pathways(save_dir, cell_type, file_name = "critical_genes_correct_up_sorted_gene.csv"):
    
    cell_type_dir = os.path.join(save_dir, cell_type)
    stats = pd.read_csv(f"{cell_type_dir}/{file_name}", sep = ',', index_col=0)
    stats = stats[stats.index.notna()]
    # Ensure the 'gene' column is of string type (if needed)
    stats['gene'] = stats['gene'].astype(str)
    
    # Remove 'p_' and 'a_a_' prefixes (if present at the start of gene names)
    stats['gene'] = stats['gene'].str.replace('^p_', '', regex=True)  # remove p_ prefix
    stats['gene'] = stats['gene'].str.replace('^a_a_', '', regex=True)  # remove a_a_ prefix
    # Optionally trim whitespace from gene names
    stats['gene'] = stats['gene'].str.strip()  # trim whitespace
    stats = stats[~stats['gene'].duplicated(keep='first')]
    stats.set_index('gene', inplace=True)
    # Print modified gene index to verify prefixes were removed
    print(stats.index)  # show first entries of the 'gene' index for inspection
    
    reactome = gmt_to_decoupler("../tutorials/gsea/c2.cp.reactome.v2025.1.Hs.symbols.gmt")

    # Retrieve resources via decoupler
    msigdb = decoupler.get_resource("MSigDB")  # contains millions of entries

    # Get reactome pathways
    reactome = msigdb.query("collection == 'reactome_pathways'")
    # Filter duplicates
    reactome = reactome[~reactome.duplicated(("geneset", "genesymbol"))]

    # Filtering genesets to match behaviour of fgsea
    geneset_size = reactome.groupby("geneset",observed=False).size()
    gsea_genesets = geneset_size.index[(geneset_size > 15) & (geneset_size < 500)]
    reactome_filtered = reactome[reactome["geneset"].isin(gsea_genesets)]
    input_genes = set(stats.index)
    print("input_genes", input_genes)
    try:
        scores, norm, pvals = decoupler.run_gsea(
            stats.T,
            reactome[reactome["geneset"].isin(gsea_genesets)],
            source="geneset",
            target="genesymbol",
            min_n=3
        )
        # Collect the genes actually used for each pathway (intersection with input genes)
        pathway_gene_dict = {}
        for pathway in gsea_genesets:
            # print('pathway', pathway)
            genes_in_pathway = set(reactome_filtered[reactome_filtered['geneset'] == pathway]['genesymbol'])
            genes_used = list(genes_in_pathway & input_genes)
            if len(genes_used) >= 3:
                pathway_gene_dict[pathway] = genes_used
        # print('pathway_gene_dict', pathway_gene_dict)
        # Step 5: Convert to DataFrame and save as CSV (one pathway per row)
        pathway_genes_df = pd.DataFrame([
            {'pathway': k, 'genes': ','.join(v), 'num_genes': len(v)}
            for k, v in pathway_gene_dict.items()
        ])

        # Step 6: Save to file
        save_path = f"{cell_type_dir}/gsea_enriched_pathway" + file_name + ".csv"  # customizable save path
        pathway_genes_df.to_csv(save_path, index=False)
        
        gsea_results = (
            pd.concat({"score": scores.T, "norm": norm.T, "pval": pvals.T}, axis=1)
            .droplevel(level=1, axis=1)
            .sort_values("pval")                    # sort by p-value
        )
        gsea_results.to_csv(f"{cell_type_dir}/gsea_results_" + file_name + ".csv")
        return gsea_results
    except:
        pass

def compute_dist_gsea_pathways(save_dir, file_name=None, save_name = None):
    # Read CSV file containing genes and cell types
    all_data = pd.read_csv(f"{save_dir}/{file_name}", sep=',')
    
    # Iterate over all cell types
    for cell_type in all_data['label'].unique():
        # Extract genes and statistics for the current cell type
        cell_type_data = all_data[all_data['label'] == cell_type]
        
        # Build stats for the current cell type (select relevant columns e.g., correlation and p_value)
        stats = cell_type_data[['genes', 'correlation', 'p_value']]
        
        # Clean gene name column
        stats['genes'] = stats['genes'].astype(str).str.replace('^p_', '', regex=True)
        stats['genes'] = stats['genes'].str.replace('^a_a_', '', regex=True).str.strip()
        stats = stats[stats['genes'].notna()]
        stats.set_index('genes', inplace=True)
        # Filter genes by correlation sign
        if save_name == "positive":
            stats = stats[stats['correlation'] > 0]
            # stats = stats[stats['correlation'] < 0]
            # Sort by correlation descending
            stats_sorted = stats.sort_values(by='correlation', ascending=False)
        elif save_name == "negative":
            stats = stats[stats['correlation'] < 0]
            # Sort by correlation ascending
            stats_sorted = stats.sort_values(by='correlation', ascending=True)
        elif "all" in save_name:
            stats_sorted = stats.sort_values(by='correlation', ascending=False)   # from high to low
        # Remove duplicate genes, keep the first occurrence
        stats_sorted = stats_sorted.loc[~stats_sorted.index.duplicated()]
        # Print current cell type and a preview of processed genes
        print(f"Processing cell type: {cell_type}, genes: {stats_sorted.index[:5]}")  # show first 5 genes
        
        # GSEA 计算
        cell_type_dir = os.path.join(save_dir, cell_type)
        os.makedirs(cell_type_dir, exist_ok=True)  # ensure cell-type folder exists
        
        # Load Reactome pathway data
        reactome = gmt_to_decoupler("../tutorials/gsea/c2.cp.reactome.v2025.1.Hs.symbols.gmt")
        msigdb = decoupler.get_resource("MSigDB")

        # Query Reactome pathways
        reactome = msigdb.query("collection == 'reactome_pathways'")
        reactome = reactome[~reactome.duplicated(("geneset", "genesymbol"))]

        # Filter genesets to sizes suitable for GSEA
        geneset_size = reactome.groupby("geneset", observed=False).size()
        gsea_genesets = geneset_size.index[(geneset_size > 15) & (geneset_size < 500)]
        reactome_filtered = reactome[reactome["geneset"].isin(gsea_genesets)]
        
        input_genes = set(stats_sorted.index)
        print("Input genes:", input_genes)
        # Print number of input genes
        print(f"Number of input genes for {cell_type}: {len(input_genes)}")
        # Perform GSEA analysis
        try:
            # scores, norm, pvals = decoupler.run_gsea(
            #     stats_sorted.T,
            #     reactome[reactome["geneset"].isin(gsea_genesets)],
            #     source="geneset",
            #     target="genesymbol",
            #     min_n=3
            # )
            scores, norm, pvals = decoupler.run_gsea(
                stats_sorted.T,
                reactome[reactome["geneset"].isin(gsea_genesets)],
                source="geneset",
                target="genesymbol",
                min_n=15
            )
            # Collect the genes actually used for each pathway (intersection with input genes)
            pathway_gene_dict = {}
            for pathway in gsea_genesets:
                genes_in_pathway = set(reactome_filtered[reactome_filtered['geneset'] == pathway]['genesymbol'])
                genes_used = list(genes_in_pathway & input_genes)
                if len(genes_used) >= 3:  # ensure at least 3 input genes per pathway
                    pathway_gene_dict[pathway] = genes_used

            # Create a DataFrame containing pathways and their genes
            pathway_genes_df = pd.DataFrame([
                {'pathway': k, 'genes': ','.join(v), 'num_genes': len(v)}
                for k, v in pathway_gene_dict.items()
            ])

            # Save to CSV file
            pathway_genes_file = os.path.join(cell_type_dir, f"gsea_dist({save_name})_enriched_pathways.csv")
            pathway_genes_df.to_csv(pathway_genes_file, index=False)
            
            # Save GSEA results
            # gsea_results = pd.concat({"score": scores.T, "norm": norm.T, "pval": pvals.T}, axis=1) \
            #     .droplevel(level=1, axis=1) \
            #     .sort_values("pval")  # sort by p-value ascending
            gsea_results = pd.DataFrame({
                'score': scores.iloc[0],      # take first row
                'norm': norm.iloc[0],
                'pval': pvals.iloc[0]
            }).sort_values("pval")
            gsea_results_file = os.path.join(cell_type_dir, f"gsea_dist({save_name})_results.csv")
            gsea_results.to_csv(gsea_results_file)

            print(f"GSEA results saved for {cell_type}")
            # return gsea_results

        except Exception as e:
            print(f"Error during GSEA computation for {cell_type}: {e}")
            continue

def plot_volin(adata, save_dir, cell_type, gene_names, save_name = "critical_genes_10_violint"):
    cell_type_dir = os.path.join(save_dir, cell_type)

    adata.var_names = adata.var_names.to_series().str.replace(r'^(a_a_)', 'a_', regex=True)
    gene_names = [g.replace('a_a_', 'a_') for g in gene_names]
    # 3. Plot using scaled version
    plt.figure(figsize=(40, 10))
    sc.pl.stacked_violin(adata, var_names=gene_names, \
                         groupby='correction',
                        )
    plt.tight_layout()
    plt.savefig(f"{cell_type_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
    
# def plot_volin(adata, gene_names):
    # cell_type_dir = os.path.join(save_dir, cell_type)
    # groupby = 'correction'

    # # 1. Extract expression matrix
    # X = adata[:, gene_names].layers["X_binned"]
    # X = X.toarray() if issparse(X) else X
    # expr_df = pd.DataFrame(X, columns=gene_names, index=adata.obs_names)

    # # 2. Add grouping information
    # expr_df[groupby] = adata.obs[groupby].values

    # # 3. Compute median expression (per group × per gene)
    # mean_expr = expr_df.groupby(groupby).mean()
    # scaled_array = MinMaxScaler().fit_transform(mean_expr)
    # # 4. Normalize medians per column (per gene)
    # scaler = MinMaxScaler()
    # mean_scaled = pd.DataFrame(
    #     scaled_array.reshape(mean_expr.shape),
    #     columns=mean_expr.columns,
    #     index=mean_expr.index
    # )
    #     # 5. Prepare plotting: x-axis is genes, y-axis is correct/incorrect
    # # 5. Reorganize plotting structure: x-axis gene, y-axis group
    # plt.figure(figsize=(1.5 * len(gene_names), 4))
    # ax = plt.gca()

    # # Construct long-format data
    # plot_df = expr_df.melt(id_vars=groupby, var_name='Gene', value_name='Expression')

    # #  gene + group compute color map
    # color_map = {
    #     (group, gene): plt.cm.Reds(mean_scaled.loc[group, gene])
    #     for group in mean_scaled.index for gene in mean_scaled.columns
    # }

    # sns.violinplot(
    #     data=plot_df,
    
    
    #     x='Gene',
    #     y=groupby,
    #     hue=None,
    #     scale='width',
    #     palette=None,
    #     inner='box',
    #     linewidth=1,
    #     ax=ax
    # )

    # # Replace each violin's facecolor (match color to gene+group mean expression)
    # for violin, (group, gene) in zip(ax.collections[::2], color_map.keys()):
    #     violin.set_facecolor(color_map[(group, gene)])
    #     violin.set_edgecolor('black')

    # ax.set_title("Violin plots (color = group mean expression, normalized globally)")
    # ax.set_ylabel("Group (correction)")
    # ax.set_xlabel("Genes")
    # plt.tight_layout()

    # save_path = f"{cell_type_dir}/critical_genes_10_violint_custom_group_by_gene.png"
    # plt.savefig(save_path, dpi=300)

    
def limit_label_length(labels, max_length=10): 
    """Break long labels into multiple lines to avoid overly long labels.""" 
    new_labels = [] 
    for label in labels: 
        if len(label) > max_length: 
            # split the label into two lines; adjust logic to split into more lines if needed
            label = label[:max_length] + '\n' + label[max_length:] 
            new_labels.append(label) 
        else:
            new_labels.append(label) 
    return new_labels

def wrap_labels_at_underscore(labels, max_length=50):
        """
        Split pathway names only at underscores to keep words intact.
        max_length: suggested maximum characters per line.
        """
        wrapped_labels = []
        for label in labels:
            if label.startswith('REACTOME_'):
                label = label[9:]  # remove the "REACTOME_" prefix (9 characters)
            if len(label) <= max_length:
                wrapped_labels.append(label)
            else:
                # split on underscores
                parts = label.split('_')    # e.g. CLASS_I_MHC_MEDIATED_ANTIGEN_PROCESSING_PRESENTATION
                lines = []
                current_line = []
                current_length = 0
                
                for part in parts:
                    # length if this part is appended (include underscore when joining)
                    part_length = len(part) + (1 if current_line else 0)  # +1 for underscore
                    
                    if current_length + part_length <= max_length:
                        current_line.append(part)
                        current_length += part_length
                    else:
                        # current line is full; start a new line
                        if current_line:
                            lines.append('_'.join(current_line))
                        current_line = [part]
                        current_length = len(part)
                
                # add the last line
                if current_line:
                    lines.append('_'.join(current_line))
                
                wrapped_labels.append('\n'.join(lines))
        
        return wrapped_labels
    
def plot_dist_correlation_gsea_all(cell_type, save_name):
    cell_type_dir = os.path.join(save_dir, cell_type)
    try:
        gsea_results_all = pd.read_csv(f"{cell_type_dir}/gsea_dist({save_name})_results.csv", sep = ',')
        all_pval_filtered = gsea_results_all[gsea_results_all['pval'] < 0.05]
        
            # ============ Top10 ES > 0 ============
        positive_top10 = (
            all_pval_filtered[all_pval_filtered["score"] > 0]
            .sort_values("score", ascending=False)
            .head(10)
            .copy()
        )

        # ============ Top10 ES < 0 ============
        negative_top10 = (
            all_pval_filtered[all_pval_filtered["score"] < 0]
            .sort_values("score", ascending=True)   # more negative scores first
            .head(10)
            .copy()
        )
            # 3. Keep only specified columns
        cols = ["source", "score", "norm", "pval"]
        positive_top10 = positive_top10[cols]
        negative_top10 = negative_top10[cols]

        # 4. Concatenate positive and negative results
        top10_pathways = pd.concat(
            [positive_top10, negative_top10],
            axis=0
        )

        # 5. Save as CSV
        output_path = os.path.join(cell_type_dir, f"top10pathway_{save_name}.csv")
        top10_pathways.to_csv(output_path, index=False)
        
        # ==================== Plotting ====================
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(f'Top 10 GSEA Results for {cell_type}', fontsize=16, fontweight='bold')

        # Process labels (use wrap_labels_at_underscore if available)
        positive_labels = wrap_labels_at_underscore(positive_top10["source"], max_length=50)
        negative_labels = wrap_labels_at_underscore(negative_top10["source"], max_length=50)

        # Positive correlation (ES > 0)
        sns.barplot(
            x=positive_top10['score'],
            y=positive_top10['source'],
            ax=axes[0],
            palette="Blues_d"
        )
        axes[0].set_title("Top 10 Pathways (Positive Correlation Genes)", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Enrichment Score (ES)", fontsize=12)
        axes[0].set_ylabel("Pathways", fontsize=12)
        axes[0].set_yticklabels(positive_labels, fontsize=11)

        # Negative correlation (ES < 0)
        sns.barplot(
            x=negative_top10['score'],
            y=negative_top10['source'],
            ax=axes[1],
            palette="Reds_d"
        )
        axes[1].set_title("Top 10 Pathways (Negative Correlation Genes)", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Enrichment Score (ES)", fontsize=12)
        axes[1].set_ylabel("Pathways", fontsize=12)
        axes[1].set_yticklabels(negative_labels, fontsize=11)

        plt.subplots_adjust(top=0.92)
        plt.tight_layout()
        plt.savefig(f"{cell_type_dir}/gsea_dist_correlation_top10_{save_name}.png", dpi=300, bbox_inches="tight")
    except Exception as e:    
        print(f"Reason: {e}")
        
def plot_dist_correlation_gsea(cell_type):
    cell_type_dir = os.path.join(save_dir, cell_type)
    try:
        gsea_results_positive = pd.read_csv(f"{cell_type_dir}/gsea_dist(positive)_results.csv", sep = ',')
        gsea_results_negative = pd.read_csv(f"{cell_type_dir}/gsea_dist(negative)_results.csv", sep = ',')
        # Filter results with p-value < 0.05
        positive_pval_filtered = gsea_results_positive[gsea_results_positive['pval'] < 0.05]
        negative_pval_filtered = gsea_results_negative[gsea_results_negative['pval'] < 0.05]
        
        # Sort and select top 10 gene sets (by p-value)
        positive_top10 = positive_pval_filtered.sort_values(by='pval', ascending=True).head(10)
        negative_top10 = negative_pval_filtered.sort_values(by='pval', ascending=True).head(10)
        
        # 3. Keep only specified columns
        cols = ["source", "score", "norm", "pval"]
        positive_top10 = positive_top10[cols]
        negative_top10 = negative_top10[cols]

        # 4. Concatenate positive and negative results
        top10_pathways = pd.concat(
            [positive_top10, negative_top10],
            axis=0
        )

        # 5. Save as CSV
        output_path = os.path.join(cell_type_dir, "top10pathway.csv")
        top10_pathways.to_csv(output_path, index=False)
        # # 创建柱状图
        # fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1行2列的图
        # fig.suptitle(f'Top 10 GSEA Results for {cell_type}', fontsize=16)
        
        # # 绘制正相关的 top 10 富集分数柱状图
        # sns.barplot(x=positive_top10['score'], y=positive_top10.index, ax=axes[0], palette="Blues_d")
        # axes[0].set_title("Top 10 Pathways (Positive Correlation Genes)")
        # axes[0].set_xlabel("Enrichment Score (ES)")
        # axes[0].set_ylabel("Pathways")
        
        # # 绘制负相关的 top 10 富集分数柱状图
        # sns.barplot(x=negative_top10['score'], y=negative_top10.index, ax=axes[1], palette="Reds_d")
        # axes[1].set_title("Top 10 Pathways (Negative Correlation Genes)")
        # axes[1].set_xlabel("Enrichment Score (ES)")
        # axes[1].set_ylabel("Pathways")
        
        # # Display/Finalize the figure
        # plt.tight_layout()
        # plt.subplots_adjust(top=0.85)  # adjust top spacing to avoid title overlap
        # plt.show()
        # plt.savefig(f"{cell_type_dir}/gsea_dist_correlation_top10.png", dpi=300, bbox_inches="tight")
        # Use function: split pathway names at underscores only

        fig, axes = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns
        fig.suptitle(f'Top 10 GSEA Results for {cell_type}', fontsize=16, fontweight='bold')
        # Process positive pathway names
        positive_labels = wrap_labels_at_underscore(positive_top10["source"], max_length=50)
        # Process negative pathway names
        negative_labels = wrap_labels_at_underscore(negative_top10["source"], max_length=50)

        # Plot positive top 10 enrichment scores
        sns.barplot(x=positive_top10['score'], y=positive_top10['source'], ax=axes[0], palette="Blues_d")
        axes[0].set_title("Top 10 Pathways (Positive Correlation Genes)", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Enrichment Score (ES)", fontsize=12)
        axes[0].set_ylabel("Pathways", fontsize=12)
        # Set y-axis labels (pathway names)
        axes[0].set_yticklabels(positive_labels, fontsize=11)

        # Plot negative top 10 enrichment scores
        sns.barplot(x=negative_top10['score'], y=negative_top10['source'], ax=axes[1], palette="Reds_d")
        axes[1].set_title("Top 10 Pathways (Negative Correlation Genes)", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Enrichment Score (ES)", fontsize=12)
        axes[1].set_ylabel("Pathways", fontsize=12)
        # Set y-axis labels (pathway names)
        axes[1].set_yticklabels(negative_labels, fontsize=11)

        # Finalize and save figure
        plt.subplots_adjust(top=0.92)  # adjust top spacing to avoid title overlap
        plt.tight_layout()
        
        plt.savefig(f"{cell_type_dir}/gsea_dist_correlation_top10.png", dpi=300, bbox_inches="tight")
    except Exception as e:    
        print(f"Reason: {e}")
        
        
def plot_gsea(cell_type):
    cell_type_dir = os.path.join(save_dir, cell_type)
    # pathways = ['REACTOME_INTERFERON_SIGNALING', 'REACTOME_SARS_COV_INFECTIONS', 'REACTOME_SARS_COV_2_INFECTION',
    # 'REACTOME_NEUTROPHIL_DEGRANULATION', 'REACTOME_SIGNALING_BY_INTERLEUKINS']
    try:
        gsea_results_correct = pd.read_csv(f"{cell_type_dir}/gsea_results_critical_genes_correct_up_sorted_gene.csv.csv", sep = ',', index_col=0)
        gsea_results_incorrect = pd.read_csv(f"{cell_type_dir}/gsea_results_critical_genes_incorrect_up_sorted_gene.csv.csv", sep = ',', index_col=0)
        correct_pval_filtered = gsea_results_correct[gsea_results_correct['pval'] < 0.05]
        incorrect_pval_filtered = gsea_results_incorrect[gsea_results_incorrect['pval'] < 0.05]

        # pathways = list(correct_pval_filtered.index) + list(incorrect_pval_filtered.index)
        # correct_list, incorrect_list = [], []
        # for i in range(len(pathways)):
        #     try:
        #         # gsea_results_RES_score = gsea_results_RES.loc[pathway]['score']   
        #         gsea_results_correct_score = gsea_results_correct.iloc[i]['score']   
        #     except:
        #         gsea_results_correct_score = 0
        #     try:
        #         # gsea_results_SEN_score = gsea_results_SEN.loc[pathway]['score']
        #         gsea_results_incorrect_score = gsea_results_incorrect.iloc[i]['score']
        #     except:
        #         gsea_results_incorrect_score = 0
        #     correct_list.append(gsea_results_correct_score)
        #     incorrect_list.append(gsea_results_incorrect_score)
        # 1. combine the two GSEA tables，and sort by the absolutely score of differentiation 
        merged_df = pd.DataFrame({
            "correct_score": gsea_results_correct.set_index(gsea_results_correct.index)["score"],
            "incorrect_score": gsea_results_incorrect.set_index(gsea_results_incorrect.index)["score"]
        })

        # Fill missing pathways (if absent in correct or incorrect)
        merged_df = merged_df.fillna(0)

        # 2. Compute absolute difference and sort
        merged_df["abs_diff"] = (merged_df["correct_score"] - merged_df["incorrect_score"]).abs()

        # 4. Sort by difference
        merged_df_sorted = merged_df.sort_values(by="abs_diff", ascending=False)

        # 5. Save the full comparison to CSV
        merged_df_sorted.to_csv(save_dir + "/pathway_scores_comparison.csv")
        
        # 3. Take top 10 pathways by absolute difference (or fewer if not available)
        top_diff_df = merged_df.sort_values(by="abs_diff", ascending=False).head(10)

        # 4. Generate lists for correct and incorrect scores
        correct_list = list(top_diff_df["correct_score"])
        incorrect_list = list(top_diff_df["incorrect_score"])
        pathways = list(top_diff_df.index)
        
        pathways_ = [p.split('REACTOME_')[1] for p in pathways]
        
        # Plot
        fig, ax = plt.subplots(figsize=(6,7))
        # fig, ax = plt.subplots(figsize=(10,6))
        # Create the barh for the two different dependent variables
        y_positions = np.arange(len(pathways_))
        ax.barh(y_positions - 0.2, correct_list, height=0.4, label='Correct', color='sandybrown')
        ax.barh(y_positions + 0.2, incorrect_list, height=0.4, label='Incorrect', color='limegreen')

        # Adding category labels
        ax.set_yticks(y_positions)
        pathways_ = limit_label_length(pathways_, max_length=70)
        ax.set_yticklabels(pathways_,fontdict={'fontsize': 12, 'fontweight': 'bold', 'family': 'Arial'})
        ax.set_xticks(np.linspace(-1,1,5))
        ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize': 12, 'family': 'Arial'})
        # Adding legend
        # ax.legend(loc = 'lower left')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f"{cell_type_dir}/gsea_examples_incorrect_correct.png",  bbox_inches='tight', dpi = 300)
    except:
        pass

def plot_dist_correlation_marker_gene_eachcelltype_violin(mean_conf, cell_type_map, sample_proto_distances, save_dir):
    # from gsea.compute_gsea import plot_volin 
    new_adata = sc.read_h5ad(f"{save_dir}/correction_adata.h5ad")
    csv_file = f"{save_dir}/dist_gene_correlation_top10_norm.csv"  # replace with your file path
    # celltype_marker_genes = convert_csv_to_dict(csv_file)
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by='correlation', ascending=False)
    # 3. Build a dictionary: each label maps to a ranked list of genes
    celltype_marker_genes = (
        df_sorted.groupby('label')['genes']
        .apply(list)
        .to_dict()
    )
    # Call plotting function for each cell type
    # gene_counts = count_marker_genes_per_celltype(celltype_marker_genes)
    # fig_size_factor = 18 / 33  # you can adjust this ratio to get an appropriate figure size
    # fig_sizes = {cell_type: (gene_counts[cell_type] * fig_size_factor + 3) for cell_type in gene_counts}
    
    for cell_type in list(cell_type_map.keys()):
        gene_names = celltype_marker_genes[cell_type]
        cell_type_adata = new_adata[new_adata.obs["celltype"] == cell_type, gene_names]
        plot_volin(cell_type_adata, save_dir, cell_type, gene_names, save_name = "dist_correlation_genes_violin")

def prototype_dist_correlation(save_dir, adata_test, cell_type_map):
    mean_conf = np.load(os.path.join(save_dir, "mean_conf.npy"))
    var_conf  = np.load(os.path.join(save_dir, "var_conf.npy"))
    is_correct_array = np.load(os.path.join(save_dir, "is_correct_array.npy"))
    # entropy_mean = np.load(os.path.join(save_dir, "entropy_mean.npy"))
    X_umap = np.load(os.path.join(save_dir, "X_umap.npy"))
    X_cell = X_umap[:mean_conf.shape[0], :]
    proto = X_umap[mean_conf.shape[0]:, :]
    
    if cell_type_map is not None:
        # index2cell = {v: k for k, v in cell_type_map.items()}

        celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()
        current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_test.obs["celltype_labels"] = current_celltype_labels
        labels = adata_test.obs["celltype_labels"]
    else:
        labels = adata_test.obs["celltype"]
        
    distances = np.linalg.norm(X_cell[:, None, :] - proto[None, :, :], axis=2)    # (30401, 17)
    # sample_proto_distances = distances[np.arange(X_cell.shape[0]), labels]    
    # exp_scores = np.exp(distances)
    # softmax_dist = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    temperature = 0.5  # or another small value
    softmax_dist = np.exp(-distances / temperature) / np.sum(np.exp(-distances / temperature), axis=1, keepdims=True)
    sample_proto_distances = softmax_dist[np.arange(X_cell.shape[0]), labels]
    plot_dist_correlation_marker_gene_eachcelltype_violin(mean_conf, cell_type_map, sample_proto_distances, save_dir)
    
if __name__=="__main__":
    import json
    from itertools import chain
    import anndata
    # save_dir = "../save/dev_BMMC-Oct23-10-16-37"
    save_dir = "../save/dev_gut-Dec08-16-46-00"
    with open(save_dir + "/celltype_to_label.json", "r") as f:
        cell_type_map = json.load(f)
        f.close()
    # new_adata = sc.read_h5ad(f"{save_dir}/correction_adata.h5ad")
    ############# GSEA for genes highly correlated with prototype distances ############
    # compute_dist_gsea_pathways(save_dir, file_name = "dist_gene_correlation_pvalue_0.05.csv", save_name = "positive")
    # compute_dist_gsea_pathways(save_dir, file_name = "dist_gene_correlation_pvalue_0.05.csv", save_name = "negative")
    compute_dist_gsea_pathways(save_dir, file_name = "dist_gene_correlation_pvalue_0.05.csv", save_name = "all_min15")
    for cell_type in list(cell_type_map.keys()):
        # plot_dist_correlation_gsea(cell_type)
        plot_dist_correlation_gsea_all(cell_type, save_name="all_min15")
        
    # for cell_type in list(cell_type_map.keys()):
    #     # cell_type = "CD4+ T activated"
    ####################################### GSEA for critical genes (correct/incorrect) & violin plots ########################################
    #     correct_gsea_results = pd.read_csv(os.path.join(save_dir, cell_type) + "/critical_genes_correct_up_sorted_gene.csv", sep = ',', index_col=0)
    #     incorrect_gsea_results = pd.read_csv(os.path.join(save_dir, cell_type) + "/critical_genes_incorrect_up_sorted_gene.csv", sep = ',', index_col=0)
    #     # compute_gsea_pathways(save_dir, cell_type, file_name = "critical_genes_incorrect_down_sorted_gene.csv")
    #     # gene_names = ["GZMA", "GZMK", "IL2RB", "CST7", "PLEK", "CCR7", "LEF1", "SCML4" , "ITK", "a_a_BCL11B"]
    #     gene_names = list(correct_gsea_results['gene'].head(5))+ list(incorrect_gsea_results['gene'].head(5))
    #     # compute_gsea_pathways(save_dir, cell_type, file_name = "critical_genes_correct_up_sorted_gene.csv")
    #     # compute_gsea_pathways(save_dir, cell_type, file_name = "critical_genes_incorrect_up_sorted_gene.csv")
    #     cell_type_adata = new_adata[new_adata.obs["celltype"] == cell_type, gene_names]
    #     if gene_names != []:
    #         plot_volin(cell_type_adata, gene_names)
    #     else:
    #         pass
        
    # for cell_type in list(cell_type_map.keys()):
    #     plot_gsea(cell_type)
    # combined_adata_test = sc.read_h5ad(f"{save_dir}/combined_adata_test.h5ad")
    # prototype_dist_correlation(save_dir, combined_adata_test, cell_type_map)
    
    