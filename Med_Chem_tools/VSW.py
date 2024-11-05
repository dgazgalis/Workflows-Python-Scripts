import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, QED
from rdkit import DataStructs
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from IPython.display import display
import scipy.stats as stats
from collections import defaultdict

# Load molecules from a file (e.g., SDF, SMILES)
def load_molecules(file_path, file_format='sdf'):
    if file_format == 'sdf':
        supplier = Chem.SDMolSupplier(file_path)
    elif file_format == 'smiles':
        supplier = Chem.SmilesMolSupplier(file_path)
    else:
        raise ValueError("Unsupported file format. Use 'sdf' or 'smiles'.")
    molecules = [mol for mol in supplier if mol is not None]
    return molecules

# Extract r_glide_XP_GScore property from molecules
def extract_gscores(molecules):
    gscores = []
    for mol in molecules:
        if mol.HasProp('r_glide_XP_GScore'):
            gscores.append(float(mol.GetProp('r_glide_XP_GScore')))
        else:
            gscores.append(None)
    return gscores

# Calculate Morgan fingerprints for a list of molecules
def calculate_fingerprints(molecules, radius=2, nBits=2048):
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits) for mol in molecules]
    return fingerprints

# Calculate Tanimoto similarity matrix
def calculate_similarity_matrix(fingerprints):
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return similarity_matrix

# Calculate chemical diversity
def calculate_diversity(similarity_matrix):
    distance_matrix = 1 - similarity_matrix
    diversity = np.mean(distance_matrix)
    return diversity

# Function to plot heatmap of the similarity matrix using matplotlib
def plot_similarity_heatmap(similarity_matrix, title="Chemical Diversity Heatmap"):
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Molecule Index")
    plt.ylabel("Molecule Index")
    plt.show()

# Function to plot hierarchical clustering dendrogram
def plot_dendrogram(similarity_matrix, title="Hierarchical Clustering Dendrogram"):
    distance_matrix = 1 - similarity_matrix
    Z = linkage(squareform(distance_matrix), 'ward')
    plt.figure(figsize=(10, 8))
    dendrogram(Z)
    plt.title(title)
    plt.xlabel("Molecule Index")
    plt.ylabel("Distance")
    plt.show()

# Function to perform clustering using AgglomerativeClustering
def cluster_fingerprints(fingerprints):
    distance_matrix = 1 - calculate_similarity_matrix(fingerprints)
    
    # Determine the optimal number of clusters using the Elbow Method
    wcss = []
    max_clusters = 10
    for n_clusters in range(1, max_clusters + 1):
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        cluster_labels = model.fit_predict(distance_matrix)
        wcss.append(sum(np.min(distance_matrix, axis=1)))
    
    # Plot the WCSS to find the elbow point
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Use the elbow point as the optimal number of clusters
    optimal_clusters = 4  # You may need to manually inspect the elbow plot and set this value

    model = AgglomerativeClustering(n_clusters=optimal_clusters, metric='precomputed', linkage='average')
    cluster_labels = model.fit_predict(distance_matrix)
    clusters = [[] for _ in range(optimal_clusters)]
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(idx)

    return clusters, cluster_labels, optimal_clusters

# Calculate physicochemical properties for a molecule
def calculate_properties(molecule):
    properties = {
        'MW': Descriptors.MolWt(molecule),
        'LogP': Descriptors.MolLogP(molecule),
        'HBA': Descriptors.NumHAcceptors(molecule),
        'HBD': Descriptors.NumHDonors(molecule),
        'TPSA': Descriptors.TPSA(molecule),
        'RotatableBonds': Descriptors.NumRotatableBonds(molecule),
        'AromaticRings': Descriptors.NumAromaticRings(molecule),
        'Fsp3': Descriptors.FractionCSP3(molecule),
        'QED': QED.qed(molecule),
        'RingCount': Descriptors.RingCount(molecule),
        'HeavyAtoms': Descriptors.HeavyAtomCount(molecule),
        'MolarRefractivity': Descriptors.MolMR(molecule)
    }
    return properties

# Function to filter molecules based on Lipinski's Rule of 5
def filter_molecules(molecules):
    filtered_molecules = []
    for mol in molecules:
        properties = calculate_properties(mol)
        if (properties['MW'] <= 500 and
            properties['LogP'] <= 5 and
            properties['HBD'] <= 5 and
            properties['HBA'] <= 10):
            filtered_molecules.append(mol)
    return filtered_molecules

# Standardize geometries for visualization
def standardize_geometries(molecules):
    for mol in molecules:
        AllChem.Compute2DCoords(mol)

# Function to visualize representative members of each cluster with properties
def visualize_cluster_representatives(molecules, clusters):
    representative_molecules = [molecules[cluster[0]] for cluster in clusters]
    standardize_geometries(representative_molecules)
    properties = [calculate_properties(mol) for mol in representative_molecules]
    legends = [
        f"Cluster {i+1}\nMW: {props['MW']:.2f}\nLogP: {props['LogP']:.2f}\nHBA: {props['HBA']}\nHBD: {props['HBD']}\nTPSA: {props['TPSA']:.2f}\nRotatable Bonds: {props['RotatableBonds']}\nAromatic Rings: {props['AromaticRings']}\nFsp3: {props['Fsp3']:.2f}\nQED: {props['QED']:.2f}\nRing Count: {props['RingCount']}\nHeavy Atoms: {props['HeavyAtoms']}\nMolar Refractivity: {props['MolarRefractivity']:.2f}"
        for i, props in enumerate(properties)
    ]
    img = Draw.MolsToGridImage(representative_molecules, molsPerRow=5, subImgSize=(300, 300), legends=legends)
    display(img)

# Function to plot histograms of key properties
def plot_property_histograms(molecules):
    properties_list = [calculate_properties(mol) for mol in molecules]
    properties_df = pd.DataFrame(properties_list)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for idx, column in enumerate(properties_df.columns):
        if idx >= len(axes):
            break
        axes[idx].hist(properties_df[column], bins=20, color='blue', alpha=0.7)
        axes[idx].set_title(f'Histogram of {column}')
        axes[idx].set_xlabel(column)
        axes[idx].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Function to plot r_glide_XP_GScore
def plot_gscores(gscores):
    plt.figure(figsize=(10, 6))
    plt.hist(gscores, bins=20, color='green', alpha=0.7)
    plt.title('Histogram of r_glide_XP_GScore')
    plt.xlabel('r_glide_XP_GScore')
    plt.ylabel('Frequency')
    plt.show()

# Function to plot PCA of the molecular fingerprints
def plot_pca(fingerprints, cluster_labels):
    pca = PCA(n_components=2)
    fingerprints_matrix = np.array([list(fp) for fp in fingerprints])
    pca_result = pca.fit_transform(fingerprints_matrix)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='tab10')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Molecular Fingerprints')
    plt.colorbar(scatter)
    plt.show()

# Function to perform statistical enrichment analysis
def perform_enrichment_analysis(molecules, clusters, properties_to_analyze):
    # Calculate properties for each molecule
    properties_list = [calculate_properties(mol) for mol in molecules]
    properties_df = pd.DataFrame(properties_list)

    # Initialize results dictionary
    enrichment_results = defaultdict(list)

    # Iterate over each property to analyze
    for property_name in properties_to_analyze:
        # Prepare contingency table for the current property
        for cluster_id, cluster in enumerate(clusters):
            cluster_molecules = [properties_df.iloc[idx][property_name] for idx in cluster]
            non_cluster_molecules = [properties_df.iloc[idx][property_name] for idx in range(len(molecules)) if idx not in cluster]

            # Example: Bin the property values (for continuous properties)
            threshold = properties_df[property_name].median()
            cluster_positive = sum(val > threshold for val in cluster_molecules)
            cluster_negative = len(cluster_molecules) - cluster_positive
            non_cluster_positive = sum(val > threshold for val in non_cluster_molecules)
            non_cluster_negative = len(non_cluster_molecules) - non_cluster_positive

            # Create contingency table
            contingency_table = [[cluster_positive, cluster_negative],
                                 [non_cluster_positive, non_cluster_negative]]

            # Perform Fisher's Exact Test
            _, p_value = stats.fisher_exact(contingency_table)
            enrichment_results[property_name].append((cluster_id, p_value))

    # Print or return the results
    for property_name, results in enrichment_results.items():
        print(f"Enrichment results for {property_name}:")
        for cluster_id, p_value in results:
            print(f"  Cluster {cluster_id}: p-value = {p_value:.4e}")

# Function to analyze gscore significance
def analyze_gscore_significance(gscores, clusters):
    gscores = np.array(gscores)
    significance_results = []

    for cluster_id, cluster in enumerate(clusters):
        cluster_gscores = gscores[cluster]
        non_cluster_gscores = np.delete(gscores, cluster)

        # Remove None values
        cluster_gscores = cluster_gscores[cluster_gscores != np.array(None)]
        non_cluster_gscores = non_cluster_gscores[non_cluster_gscores != np.array(None)]

        # Perform Mann-Whitney U test
        if len(cluster_gscores) > 0 and len(non_cluster_gscores) > 0:
            u_stat, p_value = stats.mannwhitneyu(cluster_gscores, non_cluster_gscores, alternative='two-sided')
            significance_results.append((cluster_id, p_value))

    # Print or return the results
    print("GScore Significance Results:")
    for cluster_id, p_value in significance_results:
        print(f"  Cluster {cluster_id}: p-value = {p_value:.4e}")
        
# Function to perform permutation test
def permutation_test(cluster_scores, background_scores, num_permutations=10000):
    observed_mean = np.mean(cluster_scores)
    combined = np.concatenate([cluster_scores, background_scores])
    count = 0
    
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        perm_mean = np.mean(combined[:len(cluster_scores)])
        if perm_mean >= observed_mean:
            count += 1
    
    p_value = count / num_permutations
    return p_value

# Function to compare gscore significance against statistical background
def compare_gscore_significance_with_background(gscores, clusters, num_permutations=10000):
    gscores = np.array(gscores)
    background_gscores = gscores[gscores != np.array(None)]
    significance_results = []

    for cluster_id, cluster in enumerate(clusters):
        cluster_gscores = gscores[cluster]
        cluster_gscores = cluster_gscores[cluster_gscores != np.array(None)]
        
        if len(cluster_gscores) > 0:
            p_value = permutation_test(cluster_gscores, background_gscores, num_permutations)
            significance_results.append((cluster_id, p_value))
    
    # Print or return the results
    print("GScore Significance Results Compared to Background:")
    for cluster_id, p_value in significance_results:
        print(f"  Cluster {cluster_id}: p-value = {p_value:.4e}")


# Main function to load, process, analyze, and visualize molecules
def analyze_and_visualize_chemical_diversity(file_path, file_format='sdf', radius=2, nBits=2048):
    molecules = load_molecules(file_path, file_format)
    filtered_molecules = filter_molecules(molecules)
    gscores = extract_gscores(filtered_molecules)
    fingerprints = calculate_fingerprints(filtered_molecules, radius, nBits)
    similarity_matrix = calculate_similarity_matrix(fingerprints)
    diversity = calculate_diversity(similarity_matrix)

    print(f"Chemical Diversity: {diversity:.4f}")

    plot_similarity_heatmap(similarity_matrix, title="Chemical Diversity Heatmap")
    plot_dendrogram(similarity_matrix, title="Hierarchical Clustering Dendrogram")

    clusters, cluster_labels, optimal_clusters = cluster_fingerprints(fingerprints)
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Number of clusters: {len(clusters)}")

    visualize_cluster_representatives(filtered_molecules, clusters)

    plot_property_histograms(filtered_molecules)

    plot_gscores(gscores)

    plot_pca(fingerprints, cluster_labels)

    properties_to_analyze = ['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'RotatableBonds', 'AromaticRings', 'Fsp3', 'QED', 'RingCount', 'HeavyAtoms', 'MolarRefractivity']
    perform_enrichment_analysis(filtered_molecules, clusters, properties_to_analyze)

    analyze_gscore_significance(gscores, clusters)
    compare_gscore_significance_with_background(gscores, clusters)

# Usage
if __name__ == "__main__":
    file_path = r"C:\Users\dgazg\Desktop\New folder (8)\top_500.sdf"  # Change this to your file path
    analyze_and_visualize_chemical_diversity(file_path)

