import 'package:go_router/go_router.dart';
import 'package:rec_movies_frontend/pages/create_profile_page.dart';
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
    GoRoute(
      path: '/profiles/new',
      builder: (context, state) => const CreateProfilePage(),
    ),
  ],
);
