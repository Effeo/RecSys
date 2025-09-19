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

  static Future<Map<String, dynamic>> fetchUserPreferences(
    String userId,
  ) async {
    final res = await http.get(Uri.parse('$baseUrl/users/$userId'));
    if (res.statusCode != 200) {
      throw Exception('Errore backend: ${res.statusCode}');
    }
    final data = jsonDecode(res.body) as Map<String, dynamic>;
    return (data['preferences'] as Map<String, dynamic>? ??
        <String, dynamic>{});
  }

  static Future<List<Movie>> fetchRecommendations(
    String userId, {
    bool useSampled = true, // true = sampled, false = top_deterministic
  }) async {
    final uri = Uri.parse('$baseUrl/recommendations_hybrid/$userId');
    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception('Errore backend: ${res.statusCode} — ${res.body}');
    }

    final Map<String, dynamic> data =
        jsonDecode(res.body) as Map<String, dynamic>;

    if ((data['status'] as String?) != 'ok') {
      return <Movie>[];
    }

    print(data);
    final String key = useSampled ? 'sampled' : 'top_deterministic';
    final List<dynamic> rawList =
        (data[key] as List<dynamic>? ?? const <dynamic>[]);

    final List<Movie> out = <Movie>[];
    for (final item in rawList) {
      if (item is Map<String, dynamic>) {
        final m = Movie.fromJson(item);
        // scarta righe senza titolo per evitare Movie “vuoti”
        if (m.title.trim().isNotEmpty) {
          out.add(m);
          if (out.length == 10) break; // massimizza 10
        }
      }
    }
    return out;
  }

  static Future<void> createOrUpdateUser(
    String userId,
    Map<String, dynamic> prefs,
  ) async {
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
