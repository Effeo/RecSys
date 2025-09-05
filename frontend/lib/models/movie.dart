class Movie {
  final int? movieId;
  final String title;
  final String? releaseDate; // YYYY-MM-DD
  final String? imdbUrl;
  final String? director;
  final num? runtime; // in secondi/minuti a seconda del dataset
  final num? awards;
  final num? score;       // presente nelle recommendations
  final num? similarity;  // presente nelle similar_movies

  // --- campi aggiuntivi per bandit ---
  final bool novel;              // true se marcato come "novità"
  final String? noveltyReason;   // motivo per cui è considerato nuovo
  final String? pickStrategy;    // "exploit" | "explore"

  Movie({
    required this.title,
    this.movieId,
    this.releaseDate,
    this.imdbUrl,
    this.director,
    this.runtime,
    this.awards,
    this.score,
    this.similarity,
    this.novel = false,
    this.noveltyReason,
    this.pickStrategy,
  });

  factory Movie.fromJson(Map<String, dynamic> json) {
    return Movie(
      movieId: json['movie_id'] as int?,
      title: (json['movie_title'] ?? json['title'] ?? '') as String,
      releaseDate: json['release_date'] as String?,
      imdbUrl: json['IMDb_URL'] as String?,
      director: json['director'] as String?,
      runtime: json['runtime'] as num?,
      awards: json['awards'] as num?,
      score: json['score'] as num?,
      similarity: json['similarity'] as num?,
      // --- parsing bandit ---
      novel: json['novel'] == true,
      noveltyReason: json['novelty_reason'] as String?,
      pickStrategy: json['pick_strategy'] as String?,
    );
  }
}
