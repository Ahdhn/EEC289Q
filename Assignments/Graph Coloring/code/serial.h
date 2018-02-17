/* function to check if the current color assignment is safe for vertex v */
bool IsSafe(int v, bool* graph, int V, int* color, int c)
{
	for (int i = 0; i < V; i++)
		if (graph[v * V + i] && c == color[i])
			return false;

	return true;
}

/* A recursive utility function to solve m coloring problem */
bool FindColoring(bool* graph, int V, int m, int* color, int v)
{
	/* base case: If all vertices are assigned a color then
	return true */
	if (v == V)
		return true;

	/* Consider this vertex v and try different colors */
	for (int c = 1; c <= m; c++)
	{
		/* Check if assignment of color c to v is fine*/
		if (IsSafe(v, graph, V, color, c))
		{
			color[v] = c;

			/* recur to assign colors to rest of the vertices */
			if (FindColoring(graph, V, m, color, v + 1) == true)
				return true;

			/* If assigning color c doesn't lead to a solution
			then remove it */
			color[v] = 0;
		}
	}

	/* If no color can be assigned to this vertex then return false */
	return false;
}
// Brute-force method
void GraphColoring(bool* graph, int V, int** color)
{
	*color = new int[V];

	// Initialize all color values as 0. This initialization is needed
	// correct functioning of IsSafe()
	for (int i = 0; i < V; i++)
		(*color)[i] = 0;

	for (int m = 1; m <= V; m++) {
		if (FindColoring(graph, V, m, *color, 0)) {
			break;
		}
	}
}


// Assigns colors (starting from 0) to all vertices and prints
// the assignment of colors
template <typename T>
void GreedyColoring(bool* graph, int V, T** color)
{
	using namespace std;
	*color = new T[V];
	T* solution = *color;

	// Assign the first color to first vertex
	solution[0] = 1;

	// Initialize remaining V-1 vertices as unassigned
	for (int u = 1; u < V; u++)
		solution[u] = 0;  // no color is assigned to u

	// Assign colors to remaining V-1 vertices
	for (int u = 1; u < V; u++)
	{
		set<int> used_colors; // colors already assigned to adjacent vertices
		// Process all adjacent vertices and flag their colors
		// as unavailable
		for (int i = 0; i < V; i++)
			if (graph[u * V + i])
				if (solution[i] != 0)
					used_colors.insert(solution[i]);


		// Find the first available color
		int cr;
		for (cr = 1; cr < V; cr++)
			if (used_colors.find(cr) == used_colors.end())
				break; // color not in used set

		solution[u] = cr; // Assign the found color
	}
}