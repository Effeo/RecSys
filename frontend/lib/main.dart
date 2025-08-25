import 'package:flutter/material.dart';
import 'routes.dart';

void main() {
  runApp(const RecMoviesApp());
}

class RecMoviesApp extends StatelessWidget {
  const RecMoviesApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'RecMovies',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF141414),
        colorScheme: const ColorScheme.dark(
          primary: Colors.redAccent,
          secondary: Colors.white70,
        ),
        fontFamily: 'Roboto',
      ),
      routerConfig: router,
    );
  }
}
