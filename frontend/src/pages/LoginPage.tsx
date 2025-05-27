import { useState } from "react";
import React from "react";
import { LogIn, OctagonAlert } from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL;

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({
          username: email,
          password: password,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem("token", data.access_token);
        window.location.href = "/upload";
      } else {
        if (response.status === 401) {
          setError("Incorrect email or password. Please try again.");
        } else if (response.status >= 500) {
          setError("Server error. Please try again later.");
        } else {
          setError(data.detail || "Login failed. Please check your credentials.");
        }
      }
    } catch (error) {
      setError("Unable to connect. Please check your internet connection.");
    }
    finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative flex flex-col md:flex-row items-center justify-center min-h-screen px-4 overflow-hidden">

      {/* Intro Panel */}
      <div className="relative z-10 flex-1 basis-1/3 p-6 text-center md:text-right">
        <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-pink-500 mb-4">
          Welcome to <span className="text-indigo-400">ReferAI</span>
        </h1>
        <p className="text-gray-300 text-lg">
          Log in to analyze football fouls with AI
        </p>
      </div>

      {/* Login Form */}
      <div className="relative z-10 flex basis-1/3 mr-6 max-w-sm bg-slate-800 p-8 rounded-xl shadow-lg animate-fade-in-up">
        <form
          onSubmit={handleSubmit}
          className="flex flex-col gap-4 w-full"
        >
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-200 mb-1">
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-700 text-white"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-200 mb-1">
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-700 text-white"
            />
          </div>

          {error && (
            <div className="flex items-center justify-center gap-2"><OctagonAlert className="w-5 h-5 text-red-500"/><p className="text-red-500 text-sm text-center">{error}</p></div>
          )}

          <button
            type="submit"
            disabled={loading}
            className={`w-full flex items-center justify-center gap-2 ${loading ? "bg-blue-400" : "bg-blue-600 hover:bg-blue-700"
              } text-white py-2 rounded-lg transition-colors font-semibold`}
          >
            <LogIn className="w-5 h-5" />
            {loading ? "Logging in..." : "Log in"}
          </button>

          <p className="text-sm text-gray-400 text-center mt-4">
            Don't have an account?{" "}
            <a href="/register" className="text-blue-600 hover:underline">
              Sign up
            </a>
          </p>
        </form>
      </div>
    </div>
  );
}
