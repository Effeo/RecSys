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
    );
  }
}
