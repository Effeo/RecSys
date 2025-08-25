FROM nginx:alpine

WORKDIR /app

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY build/web /usr/share/nginx/html

EXPOSE 8057

CMD ["nginx", "-g", "daemon off;"]

