import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/movie.dart';

class ApiClient {
  static const String baseUrl = 'http://127.0.0.1:8058';

  static Future<List<String>> fetchUserIds() async {
    final res = await http.get(Uri.parse('$baseUrl/users'));
    if (res.statusCode != 200) throw Exception('Errore backend: ${res.statusCode}');
    final data = jsonDecode(res.body);
    final users = data['users'];
    if (users is List) {
      if (users.isNotEmpty && users.first is Map) {
        return users.map<String>((e) => e['user_id'] as String).toList();
      } else {
        return users.cast<String>();
      }
    }
    return <String>[];
  }

  static Future<List<Movie>> fetchRecommendations(String userId) async {
    final res = await http.get(Uri.parse('$baseUrl/recommendations/$userId?top_k=20'));
    if (res.statusCode != 200) throw Exception('Errore backend: ${res.statusCode}');
    final data = jsonDecode(res.body);
    if (data['status'] == 'no_match') return [];
    final List list = data['results'] ?? [];
    return list.map((e) => Movie.fromJson(e)).toList();
  }

  static Future<List<Movie>> fetchSimilar(String title) async {
    final res = await http.get(Uri.parse('$baseUrl/similar_movies/$title?top_k=20'));
    if (res.statusCode != 200) throw Exception('Errore backend: ${res.statusCode}');
    final data = jsonDecode(res.body);
    if (data['status'] == 'no_match') return [];
    final List list = data['results'] ?? [];
    return list.map((e) => Movie.fromJson(e)).toList();
  }

  // opzionale: invio rating (predisposto)
  static Future<bool> submitRating({
    required String userId,
    required int movieId,
    required int rating, // 1..5
  }) async {
    final uri = Uri.parse('$baseUrl/rate');
    final res = await http.post(uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'user_id': userId, 'movie_id': movieId, 'rating': rating}));
    // se l'endpoint non esiste ancora, consideriamo comunque ok lato UI
    return res.statusCode == 200 || res.statusCode == 404;
  }
}
