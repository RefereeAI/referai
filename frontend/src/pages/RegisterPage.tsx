import { useState } from "react";
import { register } from "../services/auth";
import React from "react";

import { PenLine } from "lucide-react";

export default function RegisterPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrors([]);

    try {
      const { ok, data } = await register(email, password, confirmPassword);

      if (ok) {
        window.location.href = "/login";
      } else {
        throw new Error(data.detail || "Registration failed");
      }
    } catch (err: any) {
      if (err.status === 422 && err.data?.detail) {
        const messages = err.data.detail.map((d: any) =>
          d.msg.replace(/^Value error,?\s*/i, "")
        );

        setErrors(messages);
      } else if (err.data?.detail) {
        setErrors([err.data.detail]);
      } else {
        setErrors(["Registration failed. Please try again."]);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col md:flex-row items-center justify-center min-h-screen px-4">
      {/* Intro Panel */}
      <div className="flex-1 basis-1/3 p-6 text-center md:text-right">
        <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-green-600 to-blue-500 mb-4 animate-fade-in">
          Create an Account
        </h1>
        <p className="text-gray-300 text-lg">
          Join us and start your journey today.
        </p>
      </div>

      {/* Register Form */}
      <div className="flex basis-1/3 mr-6 max-w-sm bg-slate-800 p-8 rounded-xl shadow-lg">
        <form
          onSubmit={handleSubmit}
          className="flex flex-col gap-4 max-w-sm mx-auto"
        >
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-200 mb-1">
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
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
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-700 text-white"
            />
          </div>

          <div>
            <label htmlFor="confirm-password" className="block text-sm font-medium text-gray-200 mb-1">
              Confirm Password
            </label>
            <input
              id="confirm-password"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="••••••••"
              required
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-700 text-white"
            />
          </div>

          {errors.length > 0 && (
            <ul className="text-red-500 text-sm list-disc list-inside">
              {errors.map((err, i) => (
                <li key={i}>{err}</li>
              ))}
            </ul>
          )}

          <button
            type="submit"
            disabled={loading}
            className={`w-full flex items-center justify-center gap-2 ${loading ? "bg-green-400" : "bg-green-600 hover:bg-green-700"}  text-white py-2 rounded-lg transition-colors font-semibold`}
          >
            <PenLine className="w-5 h-5" />
            {loading ? "Creating account..." : "Create Account"}
          </button>

          <p className="text-sm text-gray-400 text-center mt-4">
            Already have an account?{" "}
            <a href="/login" className="text-green-600 hover:underline">
              Log in
            </a>
          </p>
        </form>
      </div>
    </div>
  );
}