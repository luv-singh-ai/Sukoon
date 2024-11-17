# multistage build image for only building the react application
# this also eliminates the risk of exposing the secrets provided via build args
FROM node:18-alpine3.17 as build

# build args for providing the envs required for building the react app
ARG SUPABASE_API_KEY
ARG SUPABASE_AUTHORIZATION_TOKEN
ARG BACKEND_ENDPOINT="https://sukoon-api.pplus.ai/"

# set a working directory
WORKDIR /app
COPY . /app

RUN rm -rf node_modules

RUN npm install

# build the react application to generate an optimized static build
# the build files will be stored in 'dist' directory
RUN VITE_SUPABASE_API_KEY=${SUPABASE_API_KEY} VITE_SUPABASE_AUTHORIZATION_TOKEN=${SUPABASE_AUTHORIZATION_TOKEN} \
  VITE_BACKEND_ENDPOINT=${BACKEND_ENDPOINT} npm run build

# use nginx base image
FROM nginx:1.21-alpine

# replace nginx configuration file
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf

# copy the build files from 'dist' directory in the builder image
COPY --from=build /app/dist /usr/share/nginx/html

# bind the server to port 80 of the container
EXPOSE 80

# start the server
CMD ["nginx", "-g", "daemon off;"]