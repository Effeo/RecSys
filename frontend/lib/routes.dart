import 'package:go_router/go_router.dart';
import 'pages/profile_selection_page.dart';
import 'pages/home_page.dart';

final router = GoRouter(
  routes: [
    GoRoute(
      path: '/',
      builder: (context, state) => const ProfileSelectionPage(),
    ),
    GoRoute(
      path: '/home/:userId',
      builder: (context, state) =>
          HomePage(userId: state.pathParameters['userId']!),
    ),
  ],
);
