// Counts the number of unique colors in a solution
#include <set>
#include <stdio.h>

template<typename T>
int CountColors(int V, T* color)
{
	using namespace std;
	int num_colors = 0;
	set<int> seen_colors;

	for (int i = 0; i < V; i++) {
		if (seen_colors.find(color[i]) == seen_colors.end()) {
			seen_colors.insert(color[i]);
			num_colors++;
		}
	}

	return num_colors;
} 

 
// Returns true if the color assignment is valid for the graph
template<typename T>
bool IsValidColoring(bool* graph, int V, T* color)
{
	for (int i = 0; i < V; i++) {
		for (int j = 0; j < V; j++) {
			if (graph[i * V + j]) {
				if (i != j && color[i] == color[j]) {
					printf("Vertex %d and Vertex %d are connected and have the same color %d\n", i, j, color[i]);
					return false;
				}
				if (color[i] < 1) {
					printf("Vertex %d has invalid color %d\n", i, color[i]);
					return false;
				}
			}
		}
	}

	return true;
}

// Returns true if the color assignment is valid for the graph
template<typename T>
bool IsValidColoring_Blocked(bool* graph, int V, T* color, uint32_t blockSize)
{
	for (int i = 0; i < V; i++) {
		for (int j = 0; j < V; j++) {
			if (graph[i * V + j]) {
				if (i != j && 
					floor(int(j)/int(blockSize)) == floor(int(i)/int(blockSize)) &&
				    color[i] == color[j]) {
					printf("Vertex %d and Vertex %d are connected and have the same color %d\n", i, j, color[i]);
					return false;
				}
				if (color[i] < 1) {
					printf("Vertex %d has invalid color %d\n", i, color[i]);
					return false;
				}
			}
		}
	}

	return true;
}

/* A utility function to print solution */
template<typename T>
void PrintSolution(T* color, int V)
{
	printf("Solution Exists:"
		" Following are the assigned colors \n");
	for (int i = 0; i < V; i++)
		printf(" %d ", color[i]);
	printf("\n");
}
