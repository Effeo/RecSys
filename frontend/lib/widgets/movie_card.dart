class Movie {
  final String title;
  final double? score;
  final double? similarity;
  final double? cfScore;
  final double? hybridScore;
  final double? prob;

  Movie({
    required this.title,
    this.score,
    this.similarity,
    this.cfScore,
    this.hybridScore,
    this.prob,
  });

  factory Movie.fromHybridJson(Map<String, dynamic> json) {
    // titolo: usa prima movie_title, poi title, altrimenti stringa vuota
    final String t = (json['movie_title'] ?? json['title'] ?? '').toString();

    return Movie(
      title: t,
      score: (json['score'] is num) ? (json['score'] as num).toDouble() : null,
      similarity: (json['similarity'] is num)
          ? (json['similarity'] as num).toDouble()
          : null,
      cfScore: (json['cf_score'] is num)
          ? (json['cf_score'] as num).toDouble()
          : null,
      hybridScore: (json['hybrid_score'] is num)
          ? (json['hybrid_score'] as num).toDouble()
          : null,
      prob: (json['probability'] is num) ? (json['probability'] as num).toDouble() : null,
    );
  }
}
