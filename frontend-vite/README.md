# Sukoon

Sukoon frontend web UI React application powered by Vite, designed for fast development and blazing-fast builds.

## Features

- ⚡️ **Vite** : Super fast build tool and development server.

- ⚛️ **React** : Build interactive UIs with ease.

- 📦 **ESM** : Supports native JavaScript modules for modern browsers.

- 🔧 **Hot Module Replacement (HMR)** : Instant updates without losing state.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- [Node.js](https://nodejs.org/)  (version 14 or later recommended)

- [npm](https://www.npmjs.com/)  or [yarn](https://yarnpkg.com/)

## Getting Started

Follow these steps to run the project locally:

### 1. Clone the repository


```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

Using npm:


```bash
npm install
```

Or with yarn:


```bash
yarn install
```

### 3. Export environment variables

Export Supabase and backend URL environment variables.

Using `.env` file:
```bash
VITE_SUPABASE_API_KEY="<API-KEY>"
VITE_SUPABASE_AUTHORIZATION_TOKEN="<AUTHORIZATION-TOKEN>"
VITE_BACKEND_ENDPOINT="https://sukoon-api.pplus.ai"
```

### 4. Start the development server

Using npm:


```bash
npm run dev
```

Or with yarn:


```bash
yarn dev
```
The app will be available at [http://localhost:5173]() .

### 5. Build for production

To create a production-ready build, run:

Using npm:


```bash
npm run build
```

Or with yarn:


```bash
yarn build
```
The built files will be available in the `dist` directory.

### 6. Preview the production build

You can preview the production build locally:

Using npm:


```bash
npm run preview
```

Or with yarn:


```bash
yarn preview
```

## Build Docker image
The Dockerfile uses multistage build images to build the React app and an Nginx server to serve it at port `80` of the container.

- The build stage is only for building the react application, which eliminates the risk of exposing the secrets provided via the build args, and then the build files are copied from it.

### 1. Build the image

You can build the image locally using the following command:

```bash
docker build --build-arg SUPABASE_API_KEY="<api-key>" --build-arg SUPABASE_AUTHORIZATION_TOKEN="<authorization-token>" -t <image-name>:<tag> .
```

### 2. Run the image

Run the image locally at port `8080`:

```bash
docker run -p 8080:80 <image-name>:<tag>
```

## Project Structure

The project structure is as follows:

```bash
src/
├── assets/         # Static assets
├── components/     # React components
├── App.jsx         # Main application component
├── index.css       # Global stylesheets
├── main.jsx        # Entry point
public/             # Static files served as-is
```

## Scripts

Here are the available yarn scripts:

- `yarn dev`: Start the development server.

- `yarn build`: Build the app for production.

- `yarn preview`: Preview the production build.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.
