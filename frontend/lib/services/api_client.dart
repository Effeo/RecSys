import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/movie.dart';

class ApiClient {
  static const String baseUrl = 'http://127.0.0.1:8058';

  static Future<List<String>> fetchUserIds() async {
    final res = await http.get(Uri.parse('$baseUrl/users'));
    if (res.statusCode != 200) {
      throw Exception('Errore backend: ${res.statusCode}');
    }
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

  // 🔹 NUOVO: recupera preferenze (per leggere liked_movie)
  static Future<Map<String, dynamic>> fetchUserPreferences(String userId) async {
    final res = await http.get(Uri.parse('$baseUrl/users/$userId'));
    if (res.statusCode != 200) {
      throw Exception('Errore backend: ${res.statusCode}');
    }
    final data = jsonDecode(res.body) as Map<String, dynamic>;
    // backend risponde con {"status":"ok","user_id":"...","preferences":{...}}
    return (data['preferences'] as Map<String, dynamic>? ?? <String, dynamic>{});
  }

  static Future<List<Movie>> fetchRecommendations(String userId) async {
    final res = await http.get(
      Uri.parse('$baseUrl/recommendations/$userId?top_k=20'),
    );
    if (res.statusCode != 200) {
      throw Exception('Errore backend: ${res.statusCode}');
    }
    final data = jsonDecode(res.body);
    if (data['status'] == 'no_match') return [];
    final List list = data['results'] ?? [];
    return list.map((e) => Movie.fromJson(e)).toList();
  }

  static Future<List<Movie>> fetchSimilar(String title) async {
    // URL-encode del titolo (spazi, caratteri speciali)
    final encodedTitle = Uri.encodeComponent(title);
    final res = await http.get(
      Uri.parse('$baseUrl/similar_movies/$encodedTitle?top_k=20'),
    );
    if (res.statusCode != 200) {
      throw Exception('Errore backend: ${res.statusCode}');
    }
    final data = jsonDecode(res.body);
    if (data['status'] == 'no_match') return [];
    final List list = data['results'] ?? [];
    return list.map((e) => Movie.fromJson(e)).toList();
  }

  // ===== Bandit (ritorna (exploit, explore)) =====
  static Future<(List<Movie>, List<Movie>)> fetchBandit(
    String userId, {
    int topK = 16,
    double epsilon = 0.35,
    int candidatePool = 140,
    int exploreExtra = 400,
    int? seed,
  }) async {
    final uri = Uri.parse(
      '$baseUrl/recommendations_bandit/$userId'
      '?top_k=$topK&epsilon=$epsilon&candidate_pool=$candidatePool&explore_extra=$exploreExtra'
      '${seed != null ? '&seed=$seed' : ''}',
    );

    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('Errore backend: ${res.statusCode}');
    }
    final data = jsonDecode(res.body);
    if (data['status'] == 'no_match') return (<Movie>[], <Movie>[]);

    final List raw = data['results'] ?? [];
    final movies = raw.map((e) => Movie.fromJson(e)).toList();

    final explore = movies.where((m) => m.pickStrategy == 'explore').toList();
    final exploit = movies.where((m) => m.pickStrategy == 'exploit').toList();

    return (exploit, explore);
  }

  static Future<void> createOrUpdateUser(
    String userId,
    Map<String, dynamic> prefs,
  ) async {
    // upsert lato backend
    final uri = Uri.parse('$baseUrl/users/$userId');
    final r = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(prefs),
    );
    if (r.statusCode != 200) {
      throw Exception(
        'Errore creazione/aggiornamento utente: HTTP ${r.statusCode} — ${r.body}',
      );
    }
  }
}
